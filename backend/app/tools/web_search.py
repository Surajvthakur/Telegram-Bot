from langchain_core.tools import tool
from tavily import TavilyClient
from app.core.config import settings

# Initialize Tavily client with API key from config
tavily_client = TavilyClient(api_key=settings.tavily_api_key)

@tool
def web_search(query: str) -> str:
    """Search the web for real-time information using Tavily."""
    try:
        response = tavily_client.search(query=query, max_results=5)
        results = response.get("results", [])

        if not results:
            return f"No search results found for: {query}"

        formatted = f"🔍 Search results for: {query}\n\n"
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No description")
            formatted += f"{i}. **{title}**\n   {content}\n   🔗 {url}\n\n"

        return formatted.strip()
    except Exception as e:
        return f"Web search failed: {str(e)}"
