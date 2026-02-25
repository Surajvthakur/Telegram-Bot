from app.core.llm import llm
from langgraph.prebuilt import create_react_agent
from app.tools.telegram import telegram

# Define tools available to the agent
tools = [] # Add tools here as they are developed

# Create the agent
app = create_react_agent(llm, tools)
