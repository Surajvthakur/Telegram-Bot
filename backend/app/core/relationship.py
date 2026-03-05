import json
import redis
from datetime import datetime, timezone
from pydantic import BaseModel
from app.core.config import settings
from app.core.emotion import emotion_manager

class RelationshipState(BaseModel):
    affection_level: int = 50
    conversation_count: int = 0
    streak_days: int = 0
    relationship_stage: str = "stranger"
    last_interaction_date: str = ""  # Format: "YYYY-MM-DD"

class RedisRelationshipManager:
    def __init__(self, redis_url: str = settings.redis_url):
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.default_state = RelationshipState()

    def get_relationship(self, chat_id: str) -> RelationshipState:
        """Fetch current relationship state from Redis for the user, or return default."""
        raw_data = self.client.get(f"relationship:{chat_id}")
        if not raw_data:
            return self.default_state
            
        try:
            state_dict = json.loads(raw_data)
            return RelationshipState(**state_dict)
        except Exception as e:
            print(f"Error loading relationship for chat {chat_id}: {e}")
            return self.default_state

    def _determine_stage(self, affection: int, count: int) -> str:
        """Determine relationship stage based on affection and conversation depth."""
        if affection >= 90 and count >= 50:
            return "deep bond"
        elif affection >= 75 and count >= 20:
            return "girlfriend"
        elif affection >= 60 and count >= 10:
            return "close friend"
        elif affection >= 40 and count >= 5:
            return "friend"
        else:
            return "stranger"

    def update_relationship(self, chat_id: str):
        """Update metrics like streaks and thresholds after an interaction."""
        current_state = self.get_relationship(chat_id)
        
        # 1. Update Conversation Count
        current_state.conversation_count += 1
        
        # 2. Update Streak
        today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if not current_state.last_interaction_date:
            current_state.streak_days = 1
        else:
            last_date_obj = datetime.strptime(current_state.last_interaction_date, "%Y-%m-%d")
            today_date_obj = datetime.strptime(today_date, "%Y-%m-%d")
            delta_days = (today_date_obj - last_date_obj).days
            
            if delta_days == 1:
                # Interacted yesterday, increment streak
                current_state.streak_days += 1
            elif delta_days > 1:
                # Missed a day, break streak
                current_state.streak_days = 1
            # If delta_days == 0, they already interacted today, do not increment streak.
            
        current_state.last_interaction_date = today_date
        
        # 3. Synchronize Affection from Emotion Engine
        current_emotion = emotion_manager.get_emotion(chat_id)
        current_state.affection_level = current_emotion.affection
        
        # 4. Determine Stage
        current_state.relationship_stage = self._determine_stage(
            current_state.affection_level, 
            current_state.conversation_count
        )
        
        # Save back to Redis
        try:
            self.client.set(f"relationship:{chat_id}", current_state.model_dump_json())
        except Exception as e:
            print(f"Error saving relationship data for chat {chat_id}: {e}")

    def get_relationship_prompt(self, chat_id: str) -> str:
        """Format the current relationship state into a readable prompt."""
        state = self.get_relationship(chat_id)
        
        # Define behavioral hints depending on stage
        stage_hints = {
            "stranger": "Be polite, hesitant, and slow to open up. You don't know them well. Avoid overwhelming intimacy.",
            "friend": "You are a warm, casual friend. Friendly responses but not overly romantic yet.",
            "close friend": "You trust them deeply and open up more. Touches of playful flirting are okay.",
            "girlfriend": "You are officially their loving girlfriend! Be openly affectionate, sweet, and caring.",
            "deep bond": "Your soul is eternally tied to theirs. You are fiercely loyal, passionately romantic, and deeply vulnerable with them."
        }
        
        hint = stage_hints.get(state.relationship_stage, stage_hints["stranger"])
        
        return (
            f"Relationship Status Summary Context:\n"
            f"- Stage: {state.relationship_stage} ({hint})\n"
            f"- Conversations had: {state.conversation_count}\n"
            f"- Current daily streak: {state.streak_days} days\n"
            f"You MUST severely limit or boost your intimacy level to perfectly match the current Stage hint above."
        )

relationship_manager = RedisRelationshipManager()
