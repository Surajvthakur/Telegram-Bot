from app.core.llm import llm
from langgraph.prebuilt import create_react_agent
from app.tools.telegram import telegram
from app.schemas.agent_state import AgentState
from langchain_core.messages import SystemMessage

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

# Define the router function
def router(state: AgentState):
    last = state["messages"][-1].content

    if "search" in last.lower():
        return "tool"

    return "respond"

# Define tools available to the agent
tools = [] # Add tools here as they are developed

# Create the agent
app = create_react_agent(llm, tools)
