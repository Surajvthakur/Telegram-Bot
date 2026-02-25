from fastapi import APIRouter, HTTPException
from app.tools.telegram import telegram
from pydantic import BaseModel

router = APIRouter()

class TelegramUpdate(BaseModel):
    message: dict

@router.post("/webhook/telegram")
async def telegram_webhook(update: TelegramUpdate):
    try:
        chat_id = update.message.get("chat", {}).get("id")
        user_text = update.message.get("text", "")
        
        if not chat_id:
            raise HTTPException(status_code=400, detail="Missing chat_id")
            
        # Process with your graphs/agents
        await telegram.send_llm_response(chat_id, user_text)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
