import sys
import os

# Suppress HuggingFace logs and progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from chatbot_graph import app
from guardrails import filter_sensitive_data
from langchain_core.messages import HumanMessage

def main():
    print("Welcome to the Parking Chatbot!")
    print("Type 'exit' to quit.")
    
    # Initialize state
    state = {
        "messages": [],
        "user_info": {},
        "dialog_stage": "general",
        "reservation_details": {}
    }
    
    while True:
        try:
            user_input = input("User: ")
        except EOFError:
            break
            
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Optional: Check input safety/PII in request if needed, or just pass through.
        
        # Invoke Graph
        state["messages"].append(HumanMessage(content=user_input))
        
        # Run graph
        result = app.invoke(state)
        
        # Get latest response
        bot_response = result["messages"][-1].content
        
        # Apply Guardrails (Output Filtering)
        safe_response = filter_sensitive_data(bot_response)
        
        print(f"Bot: {safe_response}")
        
        # Update local state tracking (optional, usually graph returns full new state)
        state = result

if __name__ == "__main__":
    main()
