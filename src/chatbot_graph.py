import os
import datetime
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.models import AgentState
from src.vector_store import get_vectorstore
from src.sql_db import get_available_spots, get_working_hours, create_reservation, check_availability, get_reservation_status

# Load environment variables if any
from dotenv import load_dotenv
load_dotenv()

# Initialize LLM
key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=key)

# Tools / Helper Functions
def get_message_text(message):
    """Safely extract text from message content (handles both string and list)."""
    if isinstance(message.content, str):
        return message.content
    elif isinstance(message.content, list):
        return " ".join([part["text"] for part in message.content if isinstance(part, dict) and "text" in part])
    return ""

def retrieve_docs_list(query: str):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)
    return docs

def retrieve_docs(query: str):
    docs = retrieve_docs_list(query)
    return "\n\n".join([d.page_content for d in docs])

def contextualize_query(state: AgentState):
    """
    Rewrite the latest user question based on conversation history.
    """
    messages = state["messages"]
    if len(messages) <= 1:
        return get_message_text(messages[-1])
        
    system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. "
        "If the user is asking about themselves (e.g. 'my name'), use the name from history if available. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    chain = prompt | llm
    
    # Simple history window
    history = messages[:-1][-4:] # Last 4 messages
    response = chain.invoke({"chat_history": history, "input": messages[-1].content})
    return response.content

def analyze_intent(state: AgentState):
    """
    Decide the next node.
    """
    messages = state["messages"]
    last_message = messages[-1]
    user_text = get_message_text(last_message).lower()
    
    # 1. Check for explicit exit from reservation
    if any(k in user_text for k in ["no", "cancel", "stop", "nevermind"]):
        return "conversation_flow"

    # 2. If we are in the middle of a reservation (using dialog_stage)
    if state.get("dialog_stage") == "reservation":
        # Unless they ask for status explicitly
        if any(k in user_text for k in ["status", "check my reservation", "is it approved"]):
            return "check_status"
        return "reservation_flow"

    # 3. Heuristic: If last bot message asked for info, we are likely trying to fill slots
    if len(messages) > 1 and isinstance(messages[-2], AIMessage):
        last_bot_msg = messages[-2].content.lower()
        if "provide" in last_bot_msg or "proceed with reservation" in last_bot_msg:
            return "reservation_flow"

    # 4. LLM Classification
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a router. Classify the user input into: "
                   "'check_status', 'reservation', 'availability', 'conversation', 'rag'."),
        ("user", "{input}")
    ])
    try:
        response = llm.invoke(classification_prompt.format_messages(input=user_text))
        intent = response.content.lower().strip()
    except Exception:
        intent = "rag"

    # 5. Keyword Overrides
    if any(k in user_text for k in ["status", "approved", "confirmed", "how's my reservation"]):
        return "check_status"
        
    if any(k in user_text for k in ["reserve", "book", "parking place", "parking spot", "reservation"]):
        return "reservation_flow"
        
    if "reservation" in intent:
        return "reservation_flow"
        
    if any(k in user_text for k in ["hours", "open", "close", "price", "cost", "rate", "available"]):
        return "dynamic_info"
        
    if "availability" in intent:
        return "dynamic_info"
        
    if "conversation" in intent or any(k in user_text for k in ["hi", "hello", "hey", "name is"]):
        return "conversation_flow"

    return "rag_flow"

# Nodes

def dynamic_info_node(state: AgentState):
    """Handle dynamic data queries"""
    hours = get_working_hours()
    spots = get_available_spots()
    free_count = len(spots)
    price_info = f"${spots[0]['price_per_hour']}/hour" if spots else "N/A"
    info = f"Working Hours: {hours}. Price: {price_info}. Currently there are {free_count} spots available."
    return {"messages": [AIMessage(content=info)]}

def conversation_node(state: AgentState):
    """Handle general conversation and personal info"""
    messages = state["messages"]
    user_info = state.get("user_info", {})
    
    # Robust extraction for Name from general conversation
    last_content = get_message_text(messages[-1])
    if not user_info.get("name"):
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract the user's name if they introduced themselves. Return JUST the name or 'null'."),
            ("human", "{input}")
        ])
        res = (extraction_prompt | llm).invoke({"input": last_content})
        extracted_name = res.content.strip()
        if extracted_name.lower() != "null" and len(extracted_name) < 50:
            user_info["name"] = extracted_name
    
    # System prompt with user context
    context_str = ""
    if user_info.get("name"):
        context_str = f"You are talking to {user_info['name']}."
    if user_info.get("car_number"):
        context_str += f" Their car plate is {user_info['car_number']}."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a helpful parking assistant. {context_str} Answer the user within the context of the chat."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    history = messages[:-1][-4:]
    response = chain.invoke({"chat_history": history, "input": last_content})
    
    return {
        "messages": [response],
        "user_info": user_info,
        "dialog_stage": "general"
    }

def rag_node(state: AgentState):
    """Handle RAG queries with history"""
    # Contextualize first
    query = contextualize_query(state)
    context_docs = retrieve_docs_list(query)
    context = "\n\n".join([d.page_content for d in context_docs])
    
    # Generate answer with safety instructions
    system_instr = (
        "You are a helpful parking assistant. Answer based on context. "
        "IMPORTANT: Do not confirm reservations here. If the user is asking about their reservation status, "
        "tell them to check their status. Do not mention names like 'John' unless they are in the context. "
        "The context is about parking rules and facts."
    )
    prompt = f"{system_instr}\n\nContext:\n{context}\n\nQuery: {query}"
    response = llm.invoke(prompt)
    return {
        "messages": [response],
        "retrieved_docs": [d.page_content for d in context_docs]
    }


def reservation_node(state: AgentState):
    """Handle reservation logic with slot filling"""
    messages = state["messages"]
    user_info = state.get("user_info", {})
    
    # 1. Try to extract info from the *latest* message if we are in reservation flow
    last_msg_content = get_message_text(messages[-1])
    
    # Only run extraction if we don't have everything yet
    # We want name, car_number, and a time/duration.
    # Simple heuristic: if we have some but not all, try to extract from latest.
    
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an extraction algorithm. Extract 'name', 'car_number', 'start_time', and 'end_time' from the input.\n"
                   "Return ONLY a JSON object with keys: 'name', 'car_number', 'start_time', 'end_time'.\n"
                   "Values should be null if not found.\n"
                   "CRITICAL: 'car_number' can be short alphanumeric strings like 'gur35' or '55y3'. Extract them exactly.\n"
                   "Handle time ranges (e.g. '8-16') by splitting into start and end.\n"
                   "Example: Input 'tomorrow 8-16' -> {{\"start_time\": \"tomorrow 8\", \"end_time\": \"tomorrow 16\", ...}}\n"),
        ("user", "{input}")
    ])
    
    chain = extraction_prompt | llm
    try:
        extraction_response = chain.invoke({"input": last_msg_content})
        content = extraction_response.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        import json
        data = json.loads(content)
        
        # PERSISTENCE: Only update if value is not null and we don't already have it (or if it's changing)
        if data.get("name") and not user_info.get("name"): 
            user_info["name"] = data["name"]
        if data.get("car_number"):
            user_info["car_number"] = data["car_number"]
        if data.get("start_time"):
            user_info["start_time"] = data["start_time"]
        if data.get("end_time"):
            user_info["end_time"] = data["end_time"]
            
    except Exception as e:
        print(f"Extraction error: {e}")

    # 2. Check what's missing
    missing = []
    if not user_info.get("name"): missing.append("name")
    if not user_info.get("car_number"): missing.append("car number")
    if not user_info.get("start_time"): missing.append("reservation time")
    
    if missing:
        return {
            "messages": [AIMessage(content=f"Please provide your {' and '.join(missing)} to proceed with reservation.")],
            "user_info": user_info,
            "dialog_stage": "reservation" # Set stage to stay in flow
        }
        
    # If we have both
    # Create reservation (Mock)
    # Pass start/end. If end is missing, pass empty string or "N/A"
    start_t = user_info.get("start_time", "N/A")
    end_t = user_info.get("end_time", "")
    
    res_msg = create_reservation(user_info["name"], user_info["car_number"], start_t, end_t)
    
    # Poll for admin decision
    import time
    while True:
        status = get_reservation_status(user_info["name"], user_info["car_number"])
        if status and status != "pending":
            break
        time.sleep(2)
        
    if status == "confirmed":
        final_msg = f"Good news! Your reservation has been CONFIRMED."
    else:
        final_msg = f"Sorry, your reservation was REJECTED by the administrator."
    
    time_display = f"{start_t}"
    if end_t:
        time_display += f" - {end_t}"
        
    return {
        "messages": [AIMessage(content=f"Request submitted for {user_info['name']} (Plate: {user_info['car_number']}, Time: {time_display}).\n{res_msg}\n\nUpdate: {final_msg}")],
        "user_info": user_info,
        "reservation_details": {"status": status},
        "dialog_stage": "general" # Reset stage after submitting request
    }

def check_status_node(state: AgentState):
    """Check status of reservation using stored user info"""
    user_info = state.get("user_info", {})
    if not user_info.get("name") or not user_info.get("car_number"):
        return {"messages": [AIMessage(content="I need your valid Name and Car Number to check the status.")]}
    
    status = get_reservation_status(user_info["name"], user_info["car_number"])
    if status is None:
        msg = "I couldn't find any reservation for you."
    else:
        msg = f"Your reservation status is: {status.upper()}"
        
    return {"messages": [AIMessage(content=msg)]}

def router_node(state: AgentState):
    intent = analyze_intent(state)
    return intent

# Graph Construction

workflow = StateGraph(AgentState)

workflow.add_node("rag_node", rag_node)
workflow.add_node("dynamic_info_node", dynamic_info_node)
workflow.add_node("reservation_node", reservation_node)
workflow.add_node("conversation_node", conversation_node)
workflow.add_node("check_status_node", check_status_node)

workflow.set_conditional_entry_point(
    router_node,
    {
        "rag_flow": "rag_node",
        "dynamic_info": "dynamic_info_node",
        "reservation_flow": "reservation_node",
        "conversation_flow": "conversation_node",
        "check_status": "check_status_node",
         "general_info": "rag_node",
         "other": "rag_node"
    }
)

workflow.add_edge("rag_node", END)
workflow.add_edge("dynamic_info_node", END)
workflow.add_edge("reservation_node", END)
workflow.add_edge("conversation_node", END)
workflow.add_edge("check_status_node", END)

app = workflow.compile()
