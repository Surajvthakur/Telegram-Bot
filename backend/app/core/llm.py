from app.core.config import settings
from groq import Groq
from langchain_groq import ChatGroq

# Raw client
groq_client = Groq(api_key=settings.groq_api_key)

# LangChain LLM for graphs/agents
llm = ChatGroq(
    groq_api_key=settings.groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
)

def get_completion(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    chat_complete_params = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
    }
    try:
        response = groq_client.chat.completions.create(**chat_complete_params)
        return response.choices.message.content
    except Exception as e:
        return f"Error: {str(e)}"
