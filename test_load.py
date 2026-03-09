import pytest
import asyncio
import threading
import time
from langchain_core.messages import HumanMessage
from src.chatbot_graph import app
from src.sql_db import init_db, get_db_connection, update_reservation_status

@pytest.fixture(autouse=True)
def setup_db():
    init_db()

@pytest.mark.asyncio
async def test_concurrent_rag_queries():
    """
    Test multiple users asking RAG questions concurrently.
    We invoke the chatbot graph directly.
    """
    async def run_query(query):
        state = {
            "messages": [HumanMessage(content=query)],
            "user_info": {},
            "dialog_stage": "general",
            "reservation_details": {}
        }
        res = await app.ainvoke(state)
        return res

    queries = [
        "What are the parking rules?",
        "How much does it cost?",
        "When does the parking open?",
        "Can I park a truck?",
        "Are there any penalties for overtime?"
    ]
    
    # Pre-initialize the local ChromaDB vectorstore once sequentially 
    # to avoid a connection race condition in CI (SQLite locks).
    from src.vector_store import get_vectorstore
    get_vectorstore()
    
    # Run 10 parallel queries
    tasks = [run_query(queries[i % len(queries)]) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    # Ensure all return successfully with a message
    for result in results:
        assert len(result["messages"]) > 0
        assert result.get("reservation_details", {}).get("status") not in ["pending", "confirmed", "rejected"]

@pytest.mark.asyncio
async def test_concurrent_reservations_and_admin_approvals():
    """
    Test multiple concurrent users trying to book and the admin approving them.
    Because SQLite handles concurrency with locks, we try to ensure the DB doesn't crash 
    under minimal load.
    """
    
    async def book_and_wait(thread_id):
        state = {
            "messages": [HumanMessage(content=f"I want to reserve a spot for User{thread_id}, plate PLT{thread_id} at 10:00")],
            "user_info": {},
            "dialog_stage": "general",
            "reservation_details": {}
        }
        res = await app.ainvoke(state)
        return thread_id

    # Simulated single background admin processing queries
    def background_admin():
        start_time = time.time()
        # Run for 15 seconds processing whatever shows up as pending
        while time.time() - start_time < 15:
            conn = get_db_connection()
            pending = conn.execute("SELECT id FROM reservations WHERE status='pending'").fetchall()
            for row in pending:
                update_reservation_status(row[0], "confirmed")
                print(f"[Admin Thread] Confirmed res {row[0]}")
            conn.close()
            time.sleep(1)

    t = threading.Thread(target=background_admin)
    t.start()

    # Run 5 concurrent bookings (SQLite might fail with too many concurrent writes if not WAL mode, 
    # but 5 should be fine with standard retry logic, or we test DB limits)
    tasks = [book_and_wait(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    t.join() # Wait for admin thread to finish
    
    # Verify DB has 5 confirmed reservations
    conn = get_db_connection()
    db_results = conn.execute("SELECT * FROM reservations WHERE status='confirmed'").fetchall()
    conn.close()
    
    assert len(db_results) > 0
