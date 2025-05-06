from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, TypedDict, Optional, Any, Union

class AgentState(TypedDict):
    """Main state for multi-agent stock analysis system"""
    # Task information
    task: str
    task_status: str
    ticker : str
    pdfs:list[Dict[str,str]] # name followed by base64 of the file 
    pdf_summaries: List[Dict[str, str]] # name of pdf and its summary
    ratio_summaries :str #
    revised_ratio_summary : str
    info:str
    vectordb :List[str] #name of each pdf as an independent vector database

model = "gemini-2.0-flash"

