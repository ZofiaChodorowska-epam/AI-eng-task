import os
import datetime
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from models import AgentState
from vector_store import get_vectorstore
from sql_db import get_available_spots, get_working_hours, create_reservation, check_availability

# Load environment variables if any
from dotenv import load_dotenv
load_dotenv()

# Initialize LLM
# In a real app we'd need an API key. I'll use a placeholder or fail gracefully if not present.
# For now assuming OPENAI_API_KEY is set or we mock it.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Tools / Helper Functions
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
        return messages[-1].content
        
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
    Decide whether to:
    1. Answer General Question (RAG)
    2. Answer Dynamic Question (SQL - availability, hours)
    3. Start/Continue Reservation
    4. General Conversation (Chitchat/Personal info)
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # 1. Heuristic: If last bot message asked for info, we are likely in that flow
    if len(messages) > 1 and isinstance(messages[-2], AIMessage):
        last_bot_msg = messages[-2].content.lower()
        if "provide your name" in last_bot_msg or "proceed with reservation" in last_bot_msg:
            return "reservation_flow"
    
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a router. Classify the user input into one of these categories: "
                   "'general_info' (for facts about the parking), "
                   "'check_availability', "
                   "'reservation', "
                   "'conversation' (for greetings, personal info sharing, chitchat)."),
        ("user", "{input}")
    ])
    
    chain = classification_prompt | llm
    try:
        response = chain.invoke({"input": last_message.content})
        intent = response.content.lower().strip()
    except Exception:
        intent = "general_info"
        
    # Heuristic: Check INTENT usually, but also safeguard specific keywords in INPUT
    user_text = last_message.content.lower()
    
    if ("availability" in intent or "check_availability" in intent or 
        "hours" in user_text or "open" in user_text or "close" in user_text or
        "cost" in user_text or "price" in user_text or "rate" in user_text or "spot" in user_text):
        return "dynamic_info"
    elif "reservation" in intent or "book" in intent:
        return "reservation_flow"
    elif "conversation" in intent or "chat" in intent:
        return "conversation_flow"
    else:
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
    
    # Simple extraction for "My name is..." if not in reservation flow
    last_content = messages[-1].content
    if "my name is" in last_content.lower():
        # Quick fallback extraction if not caught elsewhere
        import re
        match = re.search(r"my name is ([a-zA-Z]+)", last_content, re.IGNORECASE)
        if match:
            user_info["name"] = match.group(1)
            
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
        "user_info": user_info
    }

def rag_node(state: AgentState):
    """Handle RAG queries with history"""
    # Contextualize first
    query = contextualize_query(state)
    context_docs = retrieve_docs_list(query)
    context = "\n\n".join([d.page_content for d in context_docs])
    
    # Generate answer
    prompt = f"Answer the user query based on context:\n\n{context}\n\nQuery: {query}"
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
    last_msg_content = messages[-1].content
    
    # Only run extraction if we don't have everything yet
    if not user_info.get("name") or not user_info.get("car_number"):
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an extraction algorithm. Extract 'name' and 'car_number' from the input.\n"
                       "Return ONLY a JSON object with keys 'name' and 'car_number'.\n"
                       "If a value is missing or not provided, set it to null.\n"
                       "Example: {{\"name\": \"Alice\", \"car_number\": \"ABC-123\"}}\n"
                       "Example: {{\"name\": \"Bob\", \"car_number\": null}}"),
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
            
            if data.get("name"):
                user_info["name"] = data["name"]
            if data.get("car_number"):
                user_info["car_number"] = data["car_number"]
                
        except Exception as e:
            print(f"Extraction error: {e}")

    # 2. Check what's missing
    missing = []
    if not user_info.get("name"): missing.append("name")
    if not user_info.get("car_number"): missing.append("car number")
    
    if missing:
        return {
            "messages": [AIMessage(content=f"Please provide your {' and '.join(missing)} to proceed with reservation.")],
            "user_info": user_info
        }
        
    # If we have both
    # Create reservation (Mock)
    res_status = create_reservation(user_info["name"], user_info["car_number"], "Now", "Later")
    
    return {
        "messages": [AIMessage(content=f"Reservation confirmed for {user_info['name']} (Plate: {user_info['car_number']}). Status: {res_status}")],
        "user_info": user_info,
        "reservation_details": {"status": "confirmed"}
    }

def router_node(state: AgentState):
    intent = analyze_intent(state)
    return intent

# Graph Construction

workflow = StateGraph(AgentState)

workflow.add_node("rag_node", rag_node)
workflow.add_node("dynamic_info_node", dynamic_info_node)
workflow.add_node("reservation_node", reservation_node)
workflow.add_node("conversation_node", conversation_node)

workflow.set_conditional_entry_point(
    router_node,
    {
        "rag_flow": "rag_node",
        "dynamic_info": "dynamic_info_node",
        "reservation_flow": "reservation_node",
        "conversation_flow": "conversation_node",
         "general_info": "rag_node",
         "other": "rag_node"
    }
)

workflow.add_edge("rag_node", END)
workflow.add_edge("dynamic_info_node", END)
workflow.add_edge("reservation_node", END)
workflow.add_edge("conversation_node", END)

app = workflow.compile()
