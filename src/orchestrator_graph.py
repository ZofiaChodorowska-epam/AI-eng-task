from typing import TypedDict, Optional, List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import asyncio

# Import sub-components
from src.chatbot_graph import app as chatbot_app
from src.admin_graph import app as admin_app
from src.sql_db import get_pending_reservations, update_reservation_status
from src.admin_agent import log_reservation_via_mcp

class OrchestratorState(TypedDict):
    # Chatbot state
    messages: Annotated[List[BaseMessage], add_messages]
    user_info: dict
    dialog_stage: str
    reservation_details: dict
    retrieved_docs: Optional[List[str]]
    
    # Admin state
    pending_reservations: List[dict]
    current_reservation: Optional[dict]
    action: Annotated[Optional[str], lambda x, y: y] # 'approve', 'reject', 'skip'

def chatbot_node(state: OrchestratorState):
    """Run the chatbot graph."""
    state_to_pass = {
        "messages": state.get("messages", []),
        "user_info": state.get("user_info", {}),
        "dialog_stage": state.get("dialog_stage", "general"),
        "reservation_details": state.get("reservation_details", {}),
        "retrieved_docs": state.get("retrieved_docs", [])
    }
    
    # Run the chatbot graph
    result = chatbot_app.invoke(state_to_pass)
    
    return {
        "messages": result["messages"],
        "user_info": result.get("user_info", {}),
        "dialog_stage": result.get("dialog_stage", "general"),
        "reservation_details": result.get("reservation_details", {}),
        "retrieved_docs": result.get("retrieved_docs", [])
    }

def check_for_admin_escalation(state: OrchestratorState):
    """
    Check if the user just made a new reservation that requires admin attention.
    We can fetch pending reservations to see if there's any available.
    """
    pending = get_pending_reservations()
    if pending:
        # We pass the very first one for admin to review
        # Reset action to None so we don't accidentally auto-approve a new one with an old state
        return {
            "pending_reservations": pending,
            "current_reservation": pending[0],
            "action": None
        }
    return {
        "pending_reservations": [],
        "current_reservation": None,
        "action": None
    }

def admin_human_approval_node(state: OrchestratorState):
    """
    This is an interrupt point for the orchestrator.
    It pauses here for a human to review the current_reservation and set an 'action'.
    LangGraph Studio will pause execution right before this node if configured.
    """
    return state

async def process_admin_action_node(state: OrchestratorState):
    """
    Process the action set by the human (Admin).
    When resuming via Command(resume={"action": "X"}), LangGraph simply merges that into the state 
    before this node is called if we defined the state correctly. Let's make sure it reads it.
    """
    action = state.get("action")
    res = state.get("current_reservation")
    
    # Debug print to see what state has
    print(f"--- DEBUG process_admin_action_node ---")
    print(f"Action: {action}, Res: {res}")
    
    if not res:
        return {"current_reservation": None, "action": None} # Clear state if something is wrong
    
    if not action:
        # Fallback if somehow action is missing but we reached here
        return {"current_reservation": None, "action": None}

    out_state = {"current_reservation": None, "action": None}
    
    if action == 'approve':
        update_reservation_status(res['id'], 'confirmed')
        print(f"[Orchestrator] Reservation {res['id']} CONFIRMED.")
        
        # Log via MCP directly here instead of using the inner admin app
        # Because we want this orchestrator to manage the end-to-end flow.
        await log_reservation_via_mcp(res)
        
    elif action == 'reject':
        update_reservation_status(res['id'], 'rejected')
        print(f"[Orchestrator] Reservation {res['id']} REJECTED.")
        
    elif action == 'skip':
        print(f"[Orchestrator] Reservation {res['id']} SKIPPED.")
        
    return out_state

def router_after_chatbot(state: OrchestratorState):
    """Decide if we go to End or check admin escalation"""
    # If the user made a reservation in this turn, it will be added to the DB
    # We always check for pending reservations so admin can process them
    return "check_for_admin_escalation"

def router_after_admin_check(state: OrchestratorState):
    """Decide if we need human approval or we are done."""
    if state.get("current_reservation"):
        return "admin_human_approval_node"
    return END

# Build the Orchestrator Graph
builder = StateGraph(OrchestratorState)

builder.add_node("chatbot_node", chatbot_node)
builder.add_node("check_for_admin_escalation", check_for_admin_escalation)
builder.add_node("admin_human_approval_node", admin_human_approval_node)
builder.add_node("process_admin_action_node", process_admin_action_node)

builder.set_entry_point("chatbot_node")

builder.add_edge("chatbot_node", "check_for_admin_escalation")

builder.add_conditional_edges(
    "check_for_admin_escalation",
    router_after_admin_check
)

builder.add_edge("admin_human_approval_node", "process_admin_action_node")
builder.add_edge("process_admin_action_node", END) # Process one admin action per turn, then End

# Compile with interrupt for human-in-the-loop
# Note: For LangGraph Studio, we do NOT attach a MemorySaver here because Studio injects its own.
# Testing scripts or CLI runners should re-compile or supply their own saver if needed.
app = builder.compile(interrupt_before=["admin_human_approval_node"])
