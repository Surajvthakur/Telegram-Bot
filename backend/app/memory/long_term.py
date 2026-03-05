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
        
    def save_memory(self, chat_id: str, fact: str):
        """Save a new fact into the vector database with metadata."""
        memory_id = str(uuid.uuid4())
        
        metadata = {
            "user_id": str(chat_id),
            "type": "memory",
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
            
            # Extract documents from results
            documents = results.get('documents', [[]])[0]
            
            if not documents:
                return ""
            
            # Format as a bulleted list
            formatted_memories = "\n".join([f"- {doc}" for doc in documents])
            return formatted_memories
            
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

Store ONLY:
- Preferences
- Goals
- Important events
- Emotional states

AVOID saving:
- Small talk
- Greetings
- Trivial replies
- Temporary feelings
- Generic statements

Conversation:
User: {user_message}
AI: {ai_response}

If there are no valid long-term facts to extract based on the strict criteria above, return exactly the word: NONE
Otherwise, return the facts as a simple list separated by newlines, with no prefix or additional commentary.
For example:
User prefers short explanations
User wants to learn machine learning
User is actively stressed about exams
"""
        try:
            extraction_result = get_completion(prompt)
            extraction_result = extraction_result.strip()
            
            if extraction_result and extraction_result.upper() != "NONE":
                # Split by newline just in case there are multiple facts
                facts = [f.strip() for f in extraction_result.split("\n") if f.strip() and not f.strip().startswith("Here are")]
                for fact in facts:
                    # Remove common markdown bullet points if present
                    clean_fact = fact.lstrip("- *").strip()
                    if clean_fact:
                        self.save_memory(chat_id, clean_fact)
        except Exception as e:
            print(f"Error extracting long-term memory for chat {chat_id}: {e}")

vector_memory = VectorMemory()
