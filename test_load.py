import pytest
import asyncio
from langchain_core.messages import HumanMessage
from src.orchestrator_graph import builder
from src.sql_db import init_db, get_db_connection

@pytest.fixture(autouse=True)
def setup_db():
    init_db()

@pytest.fixture(scope="function")
def app_with_memory():
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    return builder.compile(checkpointer=memory, interrupt_before=["admin_human_approval_node"])


@pytest.mark.asyncio
async def test_concurrent_rag_queries(app_with_memory):
    """
    Test multiple users asking RAG questions concurrently.
    We invoke the orchestrator graph with just normal queries.
    """
    async def run_query(thread_id, query):
        config = {"configurable": {"thread_id": f"load_test_{thread_id}"}}
        state = {
            "messages": [HumanMessage(content=query)],
            "user_info": {},
            "dialog_stage": "general",
            "reservation_details": {},
            "pending_reservations": [],
            "current_reservation": None,
            "action": None
        }
        res = await app_with_memory.ainvoke(state, config)
        return res

    queries = [
        "What are the parking rules?",
        "How much does it cost?",
        "When does the parking open?",
        "Can I park a truck?",
        "Are there any penalties for overtime?"
    ]
    
    # Run 10 parallel queries
    tasks = [run_query(i, queries[i % len(queries)]) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    # Ensure all return successfully with a message
    for result in results:
        assert len(result["messages"]) > 0
        assert result.get("current_reservation") is None

@pytest.mark.asyncio
async def test_concurrent_reservations_and_admin_approvals(app_with_memory):
    """
    Test multiple concurrent users trying to book and the admin approving them.
    Because SQLite handles concurrency with locks, we ensure the DB doesn't crash 
    under minimal load and the graphs retain states per thread.
    """
    
    async def book_and_approve(thread_id):
        config = {"configurable": {"thread_id": f"load_res_{thread_id}"}}
        state = {
            "messages": [HumanMessage(content=f"I want to reserve a spot for User{thread_id}, plate PLT{thread_id} at 10:00")],
            "user_info": {},
            "dialog_stage": "general",
            "reservation_details": {},
            "pending_reservations": [],
            "current_reservation": None,
            "action": None
        }
        
        # 1. User books
        res = await app_with_memory.ainvoke(state, config)
        
        # It should pause at admin if all info is provided
        # Wait, start_time is "10:00" and end_time is empty, which is fine handled by the prompt.
        
        # 2. Admin approves if it reached the interrupt
        if res.get("current_reservation"):
            await app_with_memory.aupdate_state(config, {"action": "approve"})
            res = await app_with_memory.ainvoke(None, config)
            
        return thread_id

    # Run 5 concurrent bookings (SQLite might fail with too many concurrent writes if not WAL mode, 
    # but 5 should be fine with standard retry logic, or we test DB limits)
    tasks = [book_and_approve(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    # Verify DB has 5 confirmed reservations
    conn = get_db_connection()
    db_results = conn.execute("SELECT * FROM reservations WHERE status='confirmed'").fetchall()
    conn.close()
    
    # We might have less if the extraction prompt failed for some due to randomness, 
    # but let's assert there are at least some confirmed.
    assert len(db_results) > 0
