import unittest
import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from chatbot_graph import app
from sql_db import get_db_connection, update_reservation_status, get_reservation_status
from langchain_core.messages import HumanMessage

class TestStage2Integration(unittest.TestCase):
    def setUp(self):
        # Reset DB
        conn = get_db_connection()
        conn.execute("DELETE FROM reservations")
        conn.commit()
        conn.close()

    def test_reservation_approval_flow(self):
        print("\nTesting Stage 2: Reservation Approval Flow...")
        
        # 1. User requests reservation
        print("[User] I want to book a spot.")
        state = {
            "messages": [HumanMessage(content="I want to book a spot.")],
            "user_info": {},
            "reservation_details": {}
        }
        result = app.invoke(state)
        
        # 2. User provides details
        print("[User] My name is Bob, plate is XYZ-999, and I want to book from 10:00 to 12:00.")
        state = result
        state["messages"].append(HumanMessage(content="My name is Bob, plate is XYZ-999, and I want to book from 10:00 to 12:00."))
        result = app.invoke(state)
        last_msg = result["messages"][-1].content
        print(f"[Bot] {last_msg}")
        
        # Verify Pending and Time capture in message
        self.assertTrue(any(word in last_msg.lower() for word in ["wait", "admin", "pending", "request", "received"]), "Should mention waiting or received")
        status = get_reservation_status("Bob", "XYZ-999")
        self.assertEqual(status, "pending")
        print("[System] Status is PENDING.")
        
        # 3. User checks status (Before Approval)
        print("[User] Is my reservation approved?")
        state = result
        state["messages"].append(HumanMessage(content="Is my reservation approved?"))
        result = app.invoke(state)
        last_msg = result["messages"][-1].content
        print(f"[Bot] {last_msg}")
        self.assertIn("PENDING", last_msg)
        
        # 4. Admin Approves (Simulated)
        print("[Admin] Approving reservation...")
        # Get ID
        conn = get_db_connection()
        res_id = conn.execute("SELECT id FROM reservations WHERE name='Bob'").fetchone()[0]
        conn.close()
        update_reservation_status(res_id, "confirmed")
        
        # 5. User checks status (After Approval)
        print("[User] Check status again.")
        # Clear messages to avoid context window issues in test, or just append
        # Let's just create a new message intent to be safe on router
        state["messages"].append(HumanMessage(content="Check my reservation status."))
        result = app.invoke(state)
        last_msg = result["messages"][-1].content
        print(f"[Bot] {last_msg}")
        self.assertIn("CONFIRMED", last_msg)
        print("[System] Verification Successful!")

if __name__ == '__main__':
    unittest.main()
