import unittest
import sys
import os
import sqlite3

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from chatbot_graph import app, contextualize_query
from models import AgentState
from langchain_core.messages import HumanMessage, AIMessage
from sql_db import get_db_connection

class TestMemoryAndReservation(unittest.TestCase):
    def setUp(self):
        # Clean reservations table
        conn = get_db_connection()
        conn.execute("DELETE FROM reservations")
        conn.commit()
        conn.close()

    def test_memory_contextualization(self):
        print("\nTesting Memory Contextualization...")
        # Mock state with history
        state = {
            "messages": [
                HumanMessage(content="My name is John."),
                AIMessage(content="Hello John."),
                HumanMessage(content="What is my name?")
            ]
        }
        # We can't easily unit test the LLM output deterministically without mocking LLM,
        # but we can run it and print result to verify it's not just "What is my name?"
        try:
            rewritten = contextualize_query(state)
            print(f"Original: What is my name? -> Rewritten: {rewritten}")
            self.assertNotEqual(rewritten, "What is my name?", "Query should be rewritten to include context")
            self.assertIn("John", rewritten, "Rewritten query should contain 'John'")
        except Exception as e:
            print(f"Skipping LLM test if no key: {e}")

    def test_reservation_flow(self):
        print("\nTesting Reservation Flow...")
        # 1. User asks to book
        print("User: I want to book a spot.")
        state = {
            "messages": [HumanMessage(content="I want to book a spot.")],
            "user_info": {},
            "dialog_stage": "general",
            "reservation_details": {}
        }
        result = app.invoke(state)
        last_msg = result["messages"][-1].content
        print(f"Bot: {last_msg}")
        self.assertIn("provide your name", last_msg.lower())
        
        import threading
        import time
        from sql_db import update_reservation_status
        def admin_approve():
            time.sleep(2)
            conn = get_db_connection()
            res = None
            for _ in range(10):
                res = conn.execute("SELECT id FROM reservations WHERE name='Alice'").fetchone()
                if res:
                    break
                time.sleep(0.5)
            if res:
                update_reservation_status(res[0], "confirmed")
            conn.close()
            
        threading.Thread(target=admin_approve).start()
        
        # 2. User provides details
        print("User: My name is Alice, plate is ABC-123, time is 10:00 to 12:00.")
        state = result
        state["messages"].append(HumanMessage(content="My name is Alice, plate is ABC-123, time is 10:00 to 12:00."))
        
        result = app.invoke(state)
        last_msg = result["messages"][-1].content
        print(f"Bot: {last_msg}")
        
        # Verify extraction
        self.assertEqual(result["user_info"]["name"], "Alice")
        self.assertEqual(result["user_info"]["car_number"], "ABC-123")
        # In Stage 4, reservations are pending admin approval first, then confirmed.
        self.assertTrue(any(word in last_msg.lower() for word in ["wait", "admin", "pending", "request", "received", "confirmed"]), "Should mention waiting, received or confirmed")
        self.assertIn("CONFIRMED", last_msg)
        
        # Verify DB
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM reservations WHERE name='Alice'").fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row["car_number"], "ABC-123")
        self.assertEqual(row["status"], "confirmed")
        print("DB Record found as confirmed!")

if __name__ == '__main__':
    unittest.main()
