import os
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from src.sql_db import get_pending_reservations, update_reservation_status
from src.admin_agent import log_reservation_via_mcp
import asyncio

class AdminState(TypedDict):
    pending_reservations: List[dict]
    current_reservation: Optional[dict]
    action: Optional[str] # 'approve', 'reject', 'skip'
    message: str

def fetch_pending(state: AdminState):
    print("[AdminGraph] Fetching pending reservations...")
    pending = get_pending_reservations()
    if not pending:
        return {"pending_reservations": [], "current_reservation": None, "message": "No pending reservations found."}
    
    # Take the first one for processing
    current = pending[0]
    return {
        "pending_reservations": pending,
        "current_reservation": current,
        "message": f"Processing reservation for {current['name']} (Plate: {current['car_number']})"
    }

def human_approval(state: AdminState):
    """
    This node serves as an interrupt point in Studio.
    In the CLI, this is where we'd prompt the user.
    """
    # In LangGraph Studio, we'll interrupt before this node or use a breakpoint.
    # We can also just return and wait for an update to the 'action' field.
    return state

async def process_result(state: AdminState):
    action = state.get("action")
    res = state.get("current_reservation")
    
    if not res or not action:
        return {"message": "No action taken."}
    
    if action == 'approve':
        update_reservation_status(res['id'], 'confirmed')
        print(f"[AdminGraph] Reservation {res['id']} CONFIRMED.")
        await log_reservation_via_mcp(res)
        return {"message": f"Approved and logged {res['name']}.", "current_reservation": None, "action": None}
    elif action == 'reject':
        update_reservation_status(res['id'], 'rejected')
        print(f"[AdminGraph] Reservation {res['id']} REJECTED.")
        return {"message": f"Rejected {res['name']}.", "current_reservation": None, "action": None}
    elif action == 'skip':
        return {"message": f"Skipped {res['name']}.", "current_reservation": None, "action": None}
    
    return state

def should_continue(state: AdminState):
    if state.get("current_reservation") and not state.get("action"):
        return "human_approval"
    return "fetch_pending"

# Build Graph
builder = StateGraph(AdminState)

builder.add_node("fetch_pending", fetch_pending)
builder.add_node("human_approval", human_approval)
builder.add_node("process_result", process_result)

builder.set_entry_point("fetch_pending")

# Logic: 
# 1. Fetch -> if exists, go to human_approval
# 2. human_approval -> WAIT (interrupt) -> then process_result
# 3. process_result -> back to fetch_pending

builder.add_conditional_edges(
    "fetch_pending",
    lambda x: "human_approval" if x.get("current_reservation") else END
)

builder.add_edge("human_approval", "process_result")
builder.add_edge("process_result", "fetch_pending")

# Compile with interrupt
app = builder.compile(interrupt_before=["human_approval"])
