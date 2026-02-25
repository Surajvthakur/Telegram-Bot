from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.llm import get_completion

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = get_completion(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
