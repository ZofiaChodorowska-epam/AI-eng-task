import sys
import os
import asyncio
from langchain_core.messages import HumanMessage

# Suppress HuggingFace logs and progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.chatbot_graph import app
from src.guardrails import filter_sensitive_data

async def main():
    print("Welcome to the Parking System Chatbot!")
    print("Type 'exit' to quit.\n")
    
    # State initialization
    state = {
        "messages": [],
        "user_info": {},
        "dialog_stage": "general",
        "reservation_details": {}
    }
    
    while True:
        try:
            user_input = await asyncio.to_thread(input, "User: ")
        except EOFError:
            break
            
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Append message
        state["messages"].append(HumanMessage(content=user_input))
        
        # Invoke Graph Pipeline once with user message
        state = await app.ainvoke(state)
        
        # Get latest bot response (from the graph)
        if state.get("messages"):
            bot_response = state["messages"][-1].content
            # Apply Guardrails
            safe_response = filter_sensitive_data(bot_response)
            print(f"Bot: {safe_response}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
