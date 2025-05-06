import os
import re
import sys
import PyPDF2
import chromadb
import streamlit as st
import pandas as pd
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Genai API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Error: Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

class SimpleDocumentChunker:
    def __init__(self, collection_name: str = "documents", embedding_model: str = "models/embedding-001"):
        """
        Initialize the document chunker with ChromaDB and Google Generative AI.
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Google embedding model to use
        """
        self.embedding_model = embedding_model
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient("./chroma_db")
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            st.info(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            st.info(f"Created new collection: {collection_name}")
        
        # Text splitter for recursive chunking with optimized parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, file) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            file: Uploaded PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        try:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
        
        return text
    
    def extract_year_from_text(self, text: str, default_year: Optional[str] = None) -> str:
        """
        Extract a year from the text or use the default year.
        
        Args:
            text: Text content to search for year
            default_year: Default year to use if no year is found
            
        Returns:
            Extracted year or default year
        """
        # Look for years in the format YYYY
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        years = re.findall(year_pattern, text)
        
        if years:
            # Return the most frequent year
            return max(set(years), key=years.count)
        
        # If no year is found, use default or current year
        if default_year:
            return default_year
        else:
            return str(datetime.now().year)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using Google Generative AI.
        
        Args:
            texts: List of text chunks
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            try:
                embedding = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(embedding["embedding"])
            except Exception as e:
                st.warning(f"Error generating embedding: {e}")
                # Add a default embedding (zeros) if there's an error
                embeddings.append([0.0] * 768)  # Gemini embeddings are typically 768 dimensions
        
        return embeddings
    
    def process_document(self, file, document_title: Optional[str] = None, year: Optional[str] = None, 
                        progress_callback=None) -> List[Dict[str, Any]]:
        """
        Process a document: extract text, split into chunks,
        generate embeddings, and store in ChromaDB.
        
        Args:
            file: Uploaded PDF file
            document_title: Optional title of the document
            year: Optional year of the document
            progress_callback: Optional function to call with progress updates (0-1)
            
        Returns:
            List of processed chunks with their metadata
        """
        # Extract document title from filename if not provided
        if not document_title:
            document_title = file.name.replace('.pdf', '')
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(file)
        
        # Extract year if not provided
        if not year:
            year = self.extract_year_from_text(text)
        
        # Split text into chunks using recursive character splitter
        chunks = self.text_splitter.split_text(text)
        
        # Process each chunk
        processed_chunks = []
        embeddings = self.get_embeddings(chunks)
        
        total_chunks = len(chunks)
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create metadata for the chunk
            metadata = {
                "title": document_title,
                "year": year,
                "chunk_index": i,
                "source": file.name
            }
            
            # Add to processed chunks
            processed_chunks.append({
                "id": f"{document_title}-chunk-{i}",
                "text": chunk,
                "metadata": metadata,
                "embedding": embedding
            })
            
            # Update progress if callback provided
            if progress_callback:
                progress_value = (i + 1) / total_chunks
                progress_callback(progress_value)
        
        # Add chunks to ChromaDB
        ids = [chunk["id"] for chunk in processed_chunks]
        documents = [chunk["text"] for chunk in processed_chunks]
        metadatas = [chunk["metadata"] for chunk in processed_chunks]
        embeddings = [chunk["embedding"] for chunk in processed_chunks]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        st.success(f"Added {len(processed_chunks)} chunks to ChromaDB")
        return processed_chunks
    
    def search(self, query: str, n_results: int = 5, 
               filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks based on a query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_criteria: Optional filter criteria (e.g., {"year": "2023"})
            
        Returns:
            List of search results
        """
        # Generate embedding for the query
        query_embedding = self.get_embeddings([query])[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_criteria
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return formatted_results
    
    def filter_by_metadata(self, 
                          document_title: Optional[str] = None,
                          year: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Filter chunks by metadata criteria.
        
        Args:
            document_title: Optional document title to filter by
            year: Optional year to filter by
            
        Returns:
            List of filtered chunks
        """
        where_clause = {}
        
        if document_title:
            where_clause["title"] = document_title
        if year:
            where_clause["year"] = year
        
        results = self.collection.get(
            where=where_clause
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"])):
            formatted_results.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i]
            })
        
        return formatted_results

    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about documents in the collection.
        
        Returns:
            Dictionary with document statistics
        """
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            if not all_docs["ids"]:
                return {"total_chunks": 0}
            
            # Extract metadata
            metadatas = all_docs["metadatas"]
            
            # Get unique document titles
            titles = set([meta.get("title", "Unknown") for meta in metadatas])
            
            # Get counts by year
            year_counts = {}
            
            for meta in metadatas:
                # Count by year
                year = meta.get("year", "Unknown")
                year_counts[year] = year_counts.get(year, 0) + 1
            
            return {
                "total_chunks": len(all_docs["ids"]),
                "unique_documents": len(titles),
                "document_titles": list(titles),
                "year_counts": year_counts
            }
            
        except Exception as e:
            st.error(f"Error getting document stats: {e}")
            return {"total_chunks": 0}

# App title and configuration
st.set_page_config(
    page_title="Document Analyzer with Recursive Chunking",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "chunker" not in st.session_state:
    st.session_state.chunker = SimpleDocumentChunker(collection_name="document_collection")

if "processed_chunks" not in st.session_state:
    st.session_state.processed_chunks = []

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "upload"

# App title and description
st.title("📚 Document Analyzer with Recursive Chunking")
st.markdown("""
This app allows you to analyze PDF documents by:
1. Extracting text and splitting into chunks using recursive text chunking
2. Generating embeddings with Google Gemini
3. Storing in a vector database for semantic search
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🔍 Search & Explore", "📊 Collection Stats"])

# Tab 1: Upload and Process
with tab1:
    st.header("Upload and Process Documents")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            document_title = st.text_input("Document Title (Optional)", 
                                          value=uploaded_file.name.replace('.pdf', ''))
        
        with col2:
            current_year = str(datetime.now().year)
            document_year = st.text_input("Document Year (Optional)", 
                                         value=current_year,
                                         help="If left blank, the system will attempt to extract the year from the document text")
        
        # Process button
        if st.button("Process Document"):
            # Create a progress bar
            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0.0)
            
            # Define progress update function
            def update_progress(progress_value):
                progress_bar.progress(progress_value)
            
            with st.spinner("Processing document..."):
                # Process the document with the progress callback
                processed_chunks = st.session_state.chunker.process_document(
                    file=uploaded_file,
                    document_title=document_title,
                    year=document_year,
                    progress_callback=update_progress
                )
                
                st.session_state.processed_chunks = processed_chunks
                
                # Display success message
                st.success(f"Successfully processed {len(processed_chunks)} chunks!")
                
                # Show sample of chunks
                if processed_chunks:
                    st.subheader("Sample of Processed Chunks")
                    df = pd.DataFrame([{
                        "Chunk ID": chunk["id"],
                        "Year": chunk["metadata"]["year"],
                        "Preview": chunk["text"][:100] + "..."
                    } for chunk in processed_chunks[:5]])
                    
                    st.dataframe(df)

# Tab 2: Search and Explore
with tab2:
    st.header("Search Documents")
    
    # Search query
    query = st.text_input("Enter your search query")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_results = st.number_input("Number of results", min_value=1, max_value=20, value=5)
    
    with col2:
        # Get document stats to populate filters
        stats = st.session_state.chunker.get_document_stats()
        
        # Get unique years
        year_options = ["All"] + list(stats.get("year_counts", {}).keys())
        selected_year = st.selectbox("Filter by year", options=year_options)
    
    # Search button
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                # Prepare filter criteria
                filter_criteria = {}
                
                if selected_year != "All":
                    filter_criteria["year"] = selected_year
                
                # Perform search
                results = st.session_state.chunker.search(
                    query=query,
                    n_results=n_results,
                    filter_criteria=filter_criteria if filter_criteria else None
                )
                
                st.session_state.search_results = results
                
                # Display results
                if results:
                    st.subheader(f"Found {len(results)} relevant chunks")
                    
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1}: {result['metadata']['title']} - {result['metadata']['year']}"):
                            st.markdown(f"**Document:** {result['metadata']['title']}")
                            st.markdown(f"**Year:** {result['metadata']['year']}")
                            st.markdown(f"**Relevance Score:** {1 - result['distance']:.4f}")
                            st.markdown("**Text:**")
                            st.text(result['text'])
                else:
                    st.info("No results found. Try a different query or adjust your filters.")
        else:
            st.warning("Please enter a search query.")

# Tab 3: Collection Stats
with tab3:
    st.header("Collection Statistics")
    
    # Get document stats
    stats = st.session_state.chunker.get_document_stats()
    
    # Display basic stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Chunks", stats.get("total_chunks", 0))
    
    with col2:
        st.metric("Unique Documents", stats.get("unique_documents", 0))
    
    with col3:
        st.metric("Years Covered", len(stats.get("year_counts", {})))
    
    # Display document titles
    if stats.get("document_titles"):
        st.subheader("Documents in Collection")
        st.write(", ".join(stats.get("document_titles", [])))
    
    # Display year distribution
    st.subheader("Distribution by Year")
    year_counts = stats.get("year_counts", {})
    
    if year_counts:
        year_df = pd.DataFrame({
            "Year": list(year_counts.keys()),
            "Count": list(year_counts.values())
        }).sort_values("Year")
        
        st.line_chart(year_df, x="Year", y="Count")
    else:
        st.info("No year data available")

# Add a footer
st.markdown("---")
st.markdown("Document Analyzer with Recursive Chunking powered by Google Gemini embeddings")