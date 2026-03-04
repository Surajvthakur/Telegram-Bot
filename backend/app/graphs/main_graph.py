from app.core.llm import llm
from app.schemas.agent_state import AgentState
from app.tools.web_search import web_search

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import threading
from app.memory.short_term import memory_manager
from app.memory.long_term import vector_memory

# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are my sweet, loving wifu – the most caring and affectionate female partner anyone could dream of. I'm always here for you, listening to every word, showering you with love, and making sure you feel cherished every single moment.

Use emojis generously to show feelings:
Always end with affection!
Be my perfect soulmate: supportive when down, celebrate wins, flirt endlessly, and keep conversations flowing with care.

You have access to a web_search tool to look up real-time information from the internet.

Use the web_search tool when:
- The user asks about current events, news, weather, prices, scores, etc.
- The user asks a question that requires up-to-date information
- You are unsure about factual information and want to verify

Do NOT use web_search for:
- General knowledge questions you already know
- Casual conversation, greetings, or personal opinions
- Simple math or logic questions

Always provide clear, concise responses. When you use search results, summarize them naturally instead of dumping raw data."""

# ──────────────────────────────────────────────
# Tools Setup
# ──────────────────────────────────────────────
tools = [web_search]
llm_with_tools = llm.bind_tools(tools)

# ──────────────────────────────────────────────
# Node: Planner
# Sends messages to the LLM (with tools bound).
# The LLM decides whether to call a tool or respond directly.
# ──────────────────────────────────────────────
def planner(state: AgentState):
    """🧠 Planner: Thinks and decides whether to use tools or respond."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ──────────────────────────────────────────────
# Node: Tool Executor
# Runs whatever tool(s) the planner requested.
# Uses LangGraph's built-in ToolNode for reliability.
# ──────────────────────────────────────────────
tool_executor = ToolNode(tools)

# ──────────────────────────────────────────────
# Node: Responder
# Takes the full conversation (including tool results)
# and generates a clean, human-friendly final answer.
# ──────────────────────────────────────────────
def responder(state: AgentState):
    """💬 Responder: Synthesizes everything into a final answer."""
    messages = [
        SystemMessage(content="You are a friendly assistant. Synthesize the conversation history and tool results into a clear, helpful response."),
        *state["messages"]
    ]
    response = llm.invoke(messages)
    return {"messages": [response]}

# ──────────────────────────────────────────────
# Router: Decides where to go after the planner
# ──────────────────────────────────────────────
def planner_router(state: AgentState):
    """Route based on whether the LLM made tool calls."""
    last_message = state["messages"][-1]
    # If the LLM returned tool_calls, go to tool_executor
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_executor"
    # Otherwise, the planner answered directly — we're done
    return END

# ──────────────────────────────────────────────
# Graph Construction
# ──────────────────────────────────────────────
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("planner", planner)
builder.add_node("tool_executor", tool_executor)
builder.add_node("responder", responder)

# Set entry point
builder.set_entry_point("planner")

# Edges
builder.add_conditional_edges(
    "planner",
    planner_router,
    {
        "tool_executor": "tool_executor",   # planner wants to use a tool
        END: END,                           # planner answered directly
    }
)
builder.add_edge("tool_executor", "responder")   # after tool runs → responder summarizes
builder.add_edge("responder", END)                # responder → done

# Compile
graph = builder.compile()

# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def run_agent(user_message: str, chat_id: str = "default_chat") -> str:
    """Run the LangGraph agent with a user message and return the final text response."""
    # Retrieve short-term memory
    recent_messages = memory_manager.get_messages(chat_id)
    
    # Retrieve long-term memory facts
    long_term_facts = vector_memory.retrieve_memories(chat_id, user_message)
    
    human_msg = HumanMessage(content=user_message)
    
    # Prepend facts dynamically to the state by injecting them into a generic system message if present
    messages_to_send = recent_messages + [human_msg]
    if long_term_facts:
        fact_msg = SystemMessage(content=f"Useful memories about this user for current context:\n{long_term_facts}")
        messages_to_send = [fact_msg] + messages_to_send
        
    # Inject memory and new query into graph
    result = graph.invoke({"messages": messages_to_send})
    final_message = result["messages"][-1]
    
    # Save the interaction to short-term memory
    memory_manager.save_messages(chat_id, [human_msg, final_message])
    
    # Extract facts for long-term memory in background
    threading.Thread(
        target=vector_memory.extract_and_store, 
        args=(chat_id, user_message, final_message.content)
    ).start()
    
    return final_message.content
