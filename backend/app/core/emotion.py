import json
import redis
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.llm import get_completion

class EmotionState(BaseModel):
    mood: str = Field(default="happy")
    energy: str = Field(default="playful")
    jealousy: int = Field(default=0, ge=0, le=100)
    affection: int = Field(default=50, ge=0, le=100)

class RedisEmotionManager:
    def __init__(self, redis_url: str = settings.redis_url):
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.default_state = EmotionState()

    def get_emotion(self, chat_id: str) -> EmotionState:
        """Fetch current emotion state from Redis for the user, or return default."""
        raw_data = self.client.get(f"emotion:{chat_id}")
        if not raw_data:
            return self.default_state
            
        try:
            state_dict = json.loads(raw_data)
            return EmotionState(**state_dict)
        except Exception as e:
            print(f"Error loading emotion for chat {chat_id}: {e}")
            return self.default_state

    def get_emotion_prompt(self, chat_id: str) -> str:
        """Format the current emotional state into a readable prompt."""
        state = self.get_emotion(chat_id)
        return (
            f"Current Dynamic Emotional State:\n"
            f"- Mood: {state.mood}\n"
            f"- Energy: {state.energy}\n"
            f"- Affection Level toward user: {state.affection}/100\n"
            f"- Jealousy Level: {state.jealousy}/100\n"
            f"Let these current feelings heavily influence your tone and response right now."
        )

    def _save_emotion(self, chat_id: str, state: EmotionState):
        """Save emotion state back to Redis."""
        try:
            self.client.set(f"emotion:{chat_id}", state.model_dump_json())
        except Exception as e:
            print(f"Error saving emotion for chat {chat_id}: {e}")

    def update_emotion(self, chat_id: str, user_message: str, ai_response: str):
        """Evaluate the conversation and adjust the emotion state accordingly."""
        current_state = self.get_emotion(chat_id)
        
        prompt = f"""
You are an emotion engine calculating the internal state of a virtual girlfriend.
Analyze the latest interaction and determine the new emotional state.

Current State:
{current_state.model_dump_json()}

Interaction:
User: "{user_message}"
AI: "{ai_response}"

Rules:
- If User is ignoring, cold, or mean, decrease affection and mood might shift to sad/upset.
- If User is loving, complimenting, or engaging, increase affection (max 100) and mood might shift to happy/excited.
- If User mentions other girls, jealousy goes up.
- 'affection' and 'jealousy' must be integers between 0 and 100.
- Return ONLY valid JSON matching this schema exactly, nothing else:
{{
  "mood": "string",
  "energy": "string",
  "jealousy": integer,
  "affection": integer
}}
"""
        try:
            result = get_completion(prompt)
            # Find JSON if the model appended markdown ticks
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                result_json = json_match.group(0)
                new_state_dict = json.loads(result_json)
                new_state = EmotionState(**new_state_dict)
                self._save_emotion(chat_id, new_state)
                # print(f"[Emotion Engine] Updated state for {chat_id}: {new_state}")
            else:
                print(f"[Emotion Engine] Failed to parse JSON: {result}")
        except Exception as e:
            print(f"Error in emotion update for chat {chat_id}: {e}")

emotion_manager = RedisEmotionManager()
