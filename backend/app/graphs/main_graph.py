from app.core.llm import llm
from langgraph.prebuilt import create_react_agent
from app.tools.telegram import telegram
from app.schemas.agent_state import AgentState

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
