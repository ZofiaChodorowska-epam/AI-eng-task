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

if __name__ == '__main__':
    unittest.main()
