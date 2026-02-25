import httpx
from typing import Optional
from app.core.config import settings

class TelegramTool:
    def __init__(self):
        self.bot_token = settings.telegram_bot_token
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.client = httpx.AsyncClient()

    async def send_message(
        self, 
        chat_id: str, 
        text: str, 
        parse_mode: Optional[str] = "HTML"
    ) -> dict:
        """Send message to Telegram chat."""
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode
        }
        
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def send_llm_response(
        self, 
        chat_id: str, 
        user_message: str
    ) -> dict:
        """Send LLM-powered response to Telegram chat."""
        # Get LLM response
        llm_response = get_completion(user_message)
        
        # Send via Telegram
        return await self.send_message(chat_id, llm_response)

    async def close(self):
        await self.client.aclose()

# Global instance for easy import
telegram = TelegramTool()
