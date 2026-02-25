from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for the given query."""
    # Placeholder implementation
    return f"Search results for: {query}\n1. Groq is fast.\n2. LangGraph is powerful."
