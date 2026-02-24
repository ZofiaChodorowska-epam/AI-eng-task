import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from chatbot_graph import app
from langchain_core.messages import HumanMessage

class TestConversationMemory(unittest.TestCase):
    def test_name_memory(self):
        print("\nTesting Conversation Memory...")
        
        # 1. User introduces themselves
        state = {
            "messages": [HumanMessage(content="Hello, my name is Zofia.")],
            "user_info": {}
        }
        result = app.invoke(state)
        last_msg = result["messages"][-1].content
        print(f"Bot: {last_msg}")
        
        # Verify extraction
        self.assertEqual(result["user_info"].get("name"), "Zofia", "Name should be extracted from chitchat")
        
        # 2. User asks for their name
        state = result
        state["messages"].append(HumanMessage(content="What is my name?"))
        
        result = app.invoke(state)
        last_msg = result["messages"][-1].content
        print(f"Bot: {last_msg}")
        
        # Verify response contains name
        self.assertIn("Zofia", last_msg, "Bot should remember the name")

    def test_car_plate_memory(self):
        print("\nTesting Car Plate Memory...")
        
        # 1. User provides car plate explicitly in a booking context
        state = {
            "messages": [HumanMessage(content="I need to park. My car plate is TST999.")],
            "user_info": {}
        }
        result = app.invoke(state)
        
        # Verify extraction
        self.assertEqual(result["user_info"].get("car_number"), "TST999", "Car plate should be extracted")
        
        # 2. Cancel the reservation flow so we can talk normally
        state = result
        state["messages"].append(HumanMessage(content="cancel"))
        result = app.invoke(state)
        
        # 3. User asks for their plate with a conversational greeting to force routing
        state = result
        state["messages"].append(HumanMessage(content="Hello, what is my car plate?"))
        
        result = app.invoke(state)
        last_msg = result["messages"][-1].content
        print(f"Bot: {last_msg}")
        
        # Verify response
        self.assertIn("TST999", last_msg, "Bot should remember the car plate")

if __name__ == '__main__':
    unittest.main()
