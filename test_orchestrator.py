import pytest
import time
import threading
from langchain_core.messages import HumanMessage
from src.chatbot_graph import app
from src.sql_db import init_db, get_db_connection, update_reservation_status

@pytest.fixture(autouse=True)
def setup_db():
    init_db() # resets the database

@pytest.mark.asyncio
async def test_full_chatbot_approval_flow():
    """
    Test the complete flow:
    1. User requests a reservation.
    2. Admin approves in background thread.
    3. Chatbot returns confirmation.
    """
    state = {
        "messages": [HumanMessage(content="Hi, I am Tester with car TST123. Reserve a spot for 10:00 to 12:00")],
        "user_info": {},
        "dialog_stage": "general",
        "reservation_details": {}
    }
    
    def admin_approve():
        time.sleep(2)
        conn = get_db_connection()
        res = None
        for _ in range(10):
            res = conn.execute("SELECT id FROM reservations WHERE name='Tester'").fetchone()
            if res:
                break
            time.sleep(0.5)
        if res:
            update_reservation_status(res[0], "confirmed")
        conn.close()
        
    threading.Thread(target=admin_approve).start()
    
    # This invokes the graph which will block until the background thread approves the reservation
    result = await app.ainvoke(state)
    
    bot_msg = result["messages"][-1].content.lower()
    
    # Final response must indicate confirmed status
    assert "confirmed" in bot_msg
    
    # Ensure it's in DB as confirmed
    conn = get_db_connection()
    res = conn.execute("SELECT * FROM reservations WHERE name='Tester'").fetchone()
    conn.close()
    assert dict(res)["status"] == "confirmed"

@pytest.mark.asyncio
async def test_chatbot_rejection_flow():
    """
    Test the flow where the Admin rejects a reservation in the background thread.
    """
    state = {
        "messages": [HumanMessage(content="Hi, I am Tester2 with car TST999. Reserve a spot for 15:00 to 17:00")],
        "user_info": {},
        "dialog_stage": "general",
        "reservation_details": {}
    }
    
    def admin_reject():
        time.sleep(2)
        conn = get_db_connection()
        res = None
        for _ in range(10):
            res = conn.execute("SELECT id FROM reservations WHERE name='Tester2'").fetchone()
            if res:
                break
            time.sleep(0.5)
        if res:
            update_reservation_status(res[0], "rejected")
        conn.close()
        
    threading.Thread(target=admin_reject).start()
    
    # Invokes graph and blocks until rejection happens
    result = await app.ainvoke(state)
    
    bot_msg = result["messages"][-1].content.lower()
    
    assert "rejected" in bot_msg
    
    # Verify status changed to rejected
    conn = get_db_connection()
    res = conn.execute("SELECT * FROM reservations WHERE name='Tester2'").fetchone()
    conn.close()
    assert dict(res)["status"] == "rejected"

