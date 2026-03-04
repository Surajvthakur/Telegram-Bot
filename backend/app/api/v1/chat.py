from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.llm import get_completion

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Use a dummy chat_id for local HTTP testing
        response = get_completion(request.message) # get_completion doesn't use the graph yet
        # wait, let's fix get_completion vs run_agent in chat.py
        # Actually it uses get_completion directly. Let's redirect it to run_agent so it also gets memory.
        from app.graphs.main_graph import run_agent
        response = run_agent(request.message, chat_id="local_test_user")
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
