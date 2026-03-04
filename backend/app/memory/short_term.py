import json
import redis
from typing import List
from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict
from app.core.config import settings

class RedisMemory:
    def __init__(self, redis_url: str = settings.redis_url):
        self.client = redis.from_url(redis_url, decode_responses=True)

    def get_messages(self, chat_id: str) -> List[BaseMessage]:
        """Fetch recent messages from Redis and deserialize them."""
        raw_data = self.client.get(f"memory:{chat_id}")
        if not raw_data:
            return []
        
        try:
            dict_messages = json.loads(raw_data)
            return messages_from_dict(dict_messages)
        except Exception as e:
            print(f"Error loading memory for chat {chat_id}: {e}")
            return []

    def save_messages(self, chat_id: str, new_messages: List[BaseMessage], max_window: int = 12):
        """Append new messages and keep only the last `max_window` messages."""
        current_messages = self.get_messages(chat_id)
        
        # Add new messages
        current_messages.extend(new_messages)
        
        # Enforce sliding window
        window_messages = current_messages[-max_window:]
        
        try:
            # Serialize and save to Redis
            dict_messages = [message_to_dict(m) for m in window_messages]
            self.client.set(f"memory:{chat_id}", json.dumps(dict_messages))
        except Exception as e:
            print(f"Error saving memory for chat {chat_id}: {e}")

memory_manager = RedisMemory()
