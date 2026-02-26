from app.core.llm import llm
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from app.tools.web_search import web_search

# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful, friendly AI assistant on Telegram.
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

# Define the tools available to the agent
tools = [web_search]

# Create a ReAct agent that uses the LLM + tools
# This handles the tool-calling loop automatically
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)


def run_agent(user_message: str) -> str:
    """Run the LangGraph agent with a user message and return the final text response."""
    input_messages = {"messages": [HumanMessage(content=user_message)]}

    # Invoke the agent graph
    result = agent.invoke(input_messages)

    # Extract the final AI message
    final_message = result["messages"][-1]
    return final_message.content
