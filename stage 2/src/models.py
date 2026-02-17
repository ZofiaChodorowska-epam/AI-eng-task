from typing import Annotated, TypedDict, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class UserInfo(TypedDict):
    name: Optional[str]
    car_number: Optional[str]

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_info: UserInfo
    dialog_stage: str # "general", "reservation", "clarification"
    reservation_details: dict # start_time, end_time, etc.
    retrieved_docs: Optional[List[str]] # For evaluation purposes
