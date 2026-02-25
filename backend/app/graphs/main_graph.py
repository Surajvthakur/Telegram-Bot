from app.core.llm import llm
from langgraph.prebuilt import create_react_agent
from app.tools.telegram import telegram
from app.schemas.agent_state import AgentState
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from app.tools.web_search import web_search

# Define the planner node
def planner(state: AgentState):
    """🧠 Planner: Decides what to do."""
    messages = [
        SystemMessage(content="You are a planning agent. Decide if a tool is needed or if we can respond directly."),
        *state["messages"]
    ]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Define the responder node
def responder(state: AgentState):
    """💬 Responder: Generates human-like reply."""
    messages = [
        SystemMessage(content="You are a friendly assistant. Synthesize the history and provide a helpful response."),
        *state["messages"]
    ]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Define tools
tools = [web_search]
tool_map = {t.name: t for t in tools}

def tool_node(state: AgentState):
    """🧩 Tool Node: Executes tool calls."""
    # Simplified tool execution logic for demonstration
    # In a real app, we would use ToolNode from langgraph.prebuilt
    # but let's stick to the manual implementation as requested
    query = state["messages"][-1].content
    result = web_search.invoke(query)
    # We return the result as a message
    return {"messages": [result]}

# Define the router function
def router(state: AgentState):
    last = state["messages"][-1].content

    if "search" in last.lower():
        return "tool"

    return "respond"

# Graph construction
builder = StateGraph(AgentState)
builder.add_node("planner", planner)
builder.add_node("tool", tool_node)
builder.add_node("responder", responder)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "planner",
    router,
    {
        "tool": "tool",
        "respond": "responder"
    }
)

builder.add_edge("tool", "responder")
builder.add_edge("responder", END)

# Compile the agent
app = builder.compile()
