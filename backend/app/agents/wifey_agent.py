from app.tools.telegram import telegram

class WifeyAgent:
    async def reply_to_user(self, chat_id: str, message: str):
        await telegram.send_llm_response(chat_id, message)
