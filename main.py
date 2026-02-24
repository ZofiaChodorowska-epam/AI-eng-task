import sys
import os
import asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Suppress HuggingFace logs and progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.orchestrator_graph import builder
from src.guardrails import filter_sensitive_data

async def main():
    print("Welcome to the Parking System Orchestrator!")
    print("Type 'exit' to quit.\n")
    
    import uuid
    # Initialize config with a unique thread_id for checkpointing state (required by LangGraph interrupts)
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    # We must attach a checkpointer locally for testing in the terminal since Studio handles it differently
    memory = MemorySaver()
    app_with_memory = builder.compile(checkpointer=memory, interrupt_before=["admin_human_approval_node"])
    
    # State initialization handled by the graph implicitly
    state = {
        "messages": [],
        "user_info": {},
        "dialog_stage": "general",
        "reservation_details": {},
        "pending_reservations": [],
        "current_reservation": None,
        "action": None
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
        state = await app_with_memory.ainvoke(state, config)
        
        # Now handle potentially multiple interrupts in a row 
        # (though currently our graph only interrupts once per turn max to check admin)
        while True:
            current_state = app_with_memory.get_state(config)
            
            # If there are no next nodes to run, the graph has finished this turn
            if not current_state.next:
                break
                
            # If the next node is the human approval node, we need to prompt
            if "admin_human_approval_node" in current_state.next:
                res = current_state.values.get("current_reservation")
                if res and current_state.values.get("action") is None:
                    print(f"\n[ADMIN ALERT] New reservation pending for {res['name']} (Plate: {res['car_number']})")
                    print(f"Time: {res['start_time']} to {res['end_time']}")
                    
                    while True:
                        choice = await asyncio.to_thread(input, "Approve this reservation? (y/n/skip): ")
                        choice = choice.strip().lower()
                        
                        if choice == 'y':
                            action = 'approve'
                            break
                        elif choice == 'n':
                            action = 'reject'
                            break
                        elif choice == 'skip':
                            action = 'skip'
                            break
                        else:
                            print("Invalid input.")
                    
                    # Update state with the action directly using aupdate_state
                    await app_with_memory.aupdate_state(config, {"action": action})
                    
                    # Resume graph manually from the interrupt
                    state = await app_with_memory.ainvoke(None, config)
                else:
                    # Defensive fallback if we're paused but action was miraculously already there
                    state = await app_with_memory.ainvoke(None, config)
            else:
                # Normal graph progression if it paused for some other reason (unlikely here)
                state = await app_with_memory.ainvoke(None, config)
        
        # Finally, the graph has reached END for this turn.
        # Get latest bot response (from chatbot_node)
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
