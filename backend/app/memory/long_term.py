import os
import uuid
import datetime
import chromadb
from typing import List
from app.core.config import settings
from app.core.llm import get_completion

class VectorMemory:
    def __init__(self, persist_directory: str = "./chroma_data"):
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create the collection for long-term memory
        self.collection = self.client.get_or_create_collection(
            name="long_term_memory"
        )
        
    def save_memory(self, chat_id: str, fact: str, type_cat: str = "memory"):
        """Save a new fact into the vector database with metadata."""
        memory_id = str(uuid.uuid4())
        
        metadata = {
            "user_id": str(chat_id),
            "type": type_cat,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        try:
            self.collection.add(
                documents=[fact],
                metadatas=[metadata],
                ids=[memory_id]
            )
            print(f"[Long-Term Memory] Saved for {chat_id}: {fact}")
        except Exception as e:
            print(f"Error saving long-term memory for chat {chat_id}: {e}")

    def retrieve_memories(self, chat_id: str, query: str, limit: int = 3) -> str:
        """Query the vector database for relevant facts about the user."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where={"user_id": str(chat_id)}
            )
            
            # Extract documents and metadata from results
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            
            if not documents:
                return ""
            
            # Format as a bulleted list including the type category
            formatted_memories = []
            for doc, meta in zip(documents, metadatas):
                cat = meta.get("type", "memory") if meta else "memory"
                formatted_memories.append(f"- [{cat.upper()}] {doc}")
                
            return "\n".join(formatted_memories)
            
        except Exception as e:
            print(f"Error retrieving long-term memory for chat {chat_id}: {e}")
            return ""

    def extract_and_store(self, chat_id: str, user_message: str, ai_response: str):
        """
        Use an LLM to extract factual, long-term memory from the conversation.
        If facts exist, split and store them.
        """
        prompt = f"""
Extract any long-term factual statements about the user from the following conversation.
CRITICAL: To prevent database pollution, you must be extremely strict about what you extract.

Categorize each extracted fact into one of these strict types:
- user_profile (Name, age, location, job)
- preferences (Likes, dislikes, hobbies)
- shared_moments (Significant things you did together)
- emotions (Long-lasting emotional states or fears)
- events (Important upcoming or past life events)

AVOID saving:
- Small talk, greetings, trivial replies, temporary feelings

Conversation:
User: {user_message}
AI: {ai_response}

If there are no valid long-term facts, return exactly: NONE
Otherwise, return ONLY a valid JSON array of objects. Do not wrap in markdown tags. Example:
[
  {{"type": "user_profile", "content": "User is a software engineer"}},
  {{"type": "preferences", "content": "User loves machine learning"}}
]
"""
        try:
            extraction_result = get_completion(prompt)
            extraction_result = extraction_result.strip()
            
            if extraction_result and extraction_result.upper() != "NONE":
                import re
                import json
                
                # Extract JSON array
                json_match = re.search(r'\[.*\]', extraction_result, re.DOTALL)
                if json_match:
                    facts_array = json.loads(json_match.group(0))
                    for item in facts_array:
                        fact_type = item.get("type", "memory")
                        fact_content = item.get("content", "").strip()
                        if fact_content:
                            self.save_memory(chat_id, fact_content, fact_type)
                else:
                    print(f"[Long-Term Memory] LLM did not return a valid JSON array: {extraction_result}")
        except Exception as e:
            print(f"Error extracting long-term memory for chat {chat_id}: {e}")

vector_memory = VectorMemory()
