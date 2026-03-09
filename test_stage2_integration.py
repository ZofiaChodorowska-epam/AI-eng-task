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
        
        import threading
        def admin_approve():
            time.sleep(2)  # Give the bot time to insert the reservation
            print("[Admin] Approving reservation...")
            conn = get_db_connection()
            res = None
            for _ in range(10):
                res = conn.execute("SELECT id FROM reservations WHERE name='Bob'").fetchone()
                if res:
                    break
                time.sleep(0.5)
            if res:
                update_reservation_status(res[0], "confirmed")
            conn.close()
            
        threading.Thread(target=admin_approve).start()
        
        # 2. User provides details
        print("[User] My name is Bob, plate is XYZ-999, and I want to book from 10:00 to 12:00.")
        state = result
        state["messages"].append(HumanMessage(content="My name is Bob, plate is XYZ-999, and I want to book from 10:00 to 12:00."))
        
        # This will block until the background thread approves the reservation
        result = app.invoke(state)
        last_msg = result["messages"][-1].content
        print(f"[Bot] {last_msg}")
        
        # Verify Pending and Time capture in message
        self.assertTrue(any(word in last_msg.lower() for word in ["wait", "admin", "pending", "request", "received", "confirmed"]), "Should mention waiting, received, or confirmed")
        
        # Verify final status logic is communicated in the response
        self.assertIn("CONFIRMED", last_msg)
        status = get_reservation_status("Bob", "XYZ-999")
        self.assertEqual(status, "confirmed")
        print("[System] Verification Successful!")

    def test_missing_info_prompt(self):
        print("\nTesting Stage 2: Missing Info Prompt...")
        
        # User provides partial info
        state = {
            "messages": [HumanMessage(content="I want to book a spot for my car plate XYZ-999.")],
            "user_info": {},
            "reservation_details": {}
        }
        result = app.invoke(state)
        last_msg = result["messages"][-1].content
        
        # Verify bot asks for missing info (name and time)
        self.assertIn("name", last_msg.lower())
        self.assertIn("time", last_msg.lower())

if __name__ == '__main__':
    unittest.main()
