import pytest
from langchain_core.messages import HumanMessage
from src.orchestrator_graph import builder
from src.sql_db import init_db, get_db_connection
import asyncio

@pytest.fixture(autouse=True)
def setup_db():
    init_db() # resets the database

@pytest.mark.asyncio
async def test_full_orchestrator_flow():
    """
    Test the complete flow:
    1. User asks for availability.
    2. User requests a reservation.
    3. Orchestrator interrupts for Admin approval.
    4. Admin approves, reservation is confirmed.
    """
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    app_with_memory = builder.compile(checkpointer=memory, interrupt_before=["admin_human_approval_node"])
    
    config = {"configurable": {"thread_id": "test_orchestrator_1"}}
    
    state = {
        "messages": [HumanMessage(content="Hi, I am Tester with car TST123. Reserve a spot for 10:00 to 12:00")],
        "user_info": {},
        "dialog_stage": "general",
        "reservation_details": {},
        "pending_reservations": [],
        "current_reservation": None,
        "action": None
    }
    
    result = await app_with_memory.ainvoke(state, config)
    
    # 1. State should be paused at the admin check if the reservation was created.
    # The chatbot might need two turns? Let's check status.
    assert "messages" in result
    
    bot_msg = result["messages"][-1].content.lower()
    print("Bot says:", bot_msg)
    
    # It might ask to confirm "start_time" or it created the reservation.
    # If it created the reservation, current_reservation should be set.
    # If not, let's provide missing info
    if not result.get("current_reservation"):
        # Let's see if we need another turn.
        state = result
        state["messages"].append(HumanMessage(content="My name is Tester, car TST123, time 10:00 to 12:00"))
        result = await app_with_memory.ainvoke(state, config)

    # Now it should be interrupted
    assert result.get("current_reservation") is not None
    assert result["current_reservation"]["name"] == "Tester"
    
    # Ensure it's in DB as pending
    conn = get_db_connection()
    res = conn.execute("SELECT * FROM reservations WHERE name='Tester'").fetchone()
    conn.close()
    assert dict(res)["status"] == "pending"
    
    from langgraph.types import Command
    
    # 2. Admin approves
    # Update the graph state directly using the checkpoint configuration
    await app_with_memory.aupdate_state(config, {"action": "approve"})
    
    # Resume graph execution (pass None since we just updated the state)
    result = await app_with_memory.ainvoke(None, config)
    
    # 3. Verify status changed to confirmed
    conn = get_db_connection()
    res = conn.execute("SELECT * FROM reservations WHERE name='Tester'").fetchone()
    conn.close()
    assert dict(res)["status"] == "confirmed"
    
    # Final state should have action cleared and current_reservation cleared due to process_admin_action_node logic
    assert result.get("action") is None
    assert result.get("current_reservation") is None


@pytest.mark.asyncio
async def test_orchestrator_rejection_flow():
    """
    Test the flow where the Admin rejects a reservation.
    """
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    app_with_memory = builder.compile(checkpointer=memory, interrupt_before=["admin_human_approval_node"])
    
    config = {"configurable": {"thread_id": "test_orchestrator_reject"}}
    
    state = {
        "messages": [HumanMessage(content="Hi, I am Tester2 with car TST999. Reserve a spot for 15:00 to 17:00")],
        "user_info": {},
        "dialog_stage": "general",
        "reservation_details": {},
        "pending_reservations": [],
        "current_reservation": None,
        "action": None
    }
    
    result = await app_with_memory.ainvoke(state, config)
    
    # Ensure it's interrupted
    assert result.get("current_reservation") is not None
    assert result["current_reservation"]["name"] == "Tester2"
    
    # Verify in DB as pending
    conn = get_db_connection()
    res = conn.execute("SELECT * FROM reservations WHERE name='Tester2'").fetchone()
    conn.close()
    assert dict(res)["status"] == "pending"
    
    # Admin rejects
    await app_with_memory.aupdate_state(config, {"action": "reject"})
    result = await app_with_memory.ainvoke(None, config)
    
    # Verify status changed to rejected
    conn = get_db_connection()
    res = conn.execute("SELECT * FROM reservations WHERE name='Tester2'").fetchone()
    conn.close()
    assert dict(res)["status"] == "rejected"
    
    # Final state should be cleared
    assert result.get("action") is None
    assert result.get("current_reservation") is None
