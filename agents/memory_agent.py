"""
Memory Agent for managing short-term and long-term memory
"""
import asyncio
import json
import redis
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import settings

logger = logging.getLogger(__name__)


class MemoryAgent:
    """Handles both short-term and long-term memory management"""
    
    def __init__(self):
        # Redis for short-term memory (session state)
        self.redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        
        # Qdrant for long-term memory (persistent storage)
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=settings.google_api_key,
            model=settings.embedding_model
        )
        
        self.memory_collection = "conversation_memory"
        self._ensure_memory_collection_exists()
    
    def _ensure_memory_collection_exists(self):
        """Create memory collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections()
            if self.memory_collection not in [c.name for c in collections.collections]:
                self.qdrant_client.create_collection(
                    collection_name=self.memory_collection,
                    vectors_config=VectorParams(
                        size=3072,  # Gemini gemini-embedding-001 embedding size
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created memory collection: {self.memory_collection}")
        except Exception as e:
            logger.error(f"Error ensuring memory collection exists: {e}")
    
    async def store_conversation_turn(
        self,
        session_id: str,
        user_query: str,
        bot_response: str,
        context: Dict[str, Any] = None
    ) -> bool:
        """Store a conversation turn in both short and long-term memory"""
        try:
            # Store in short-term memory (Redis)
            await self._store_short_term_memory(session_id, user_query, bot_response, context)
            
            # Store in long-term memory (Qdrant) if significant
            if await self._is_significant_conversation(user_query, bot_response):
                await self._store_long_term_memory(session_id, user_query, bot_response, context)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing conversation turn: {e}")
            return False
    
    async def _store_short_term_memory(
        self,
        session_id: str,
        user_query: str,
        bot_response: str,
        context: Dict[str, Any]
    ):
        """Store conversation in Redis for current session"""
        try:
            conversation_key = f"session:{session_id}:conversation"
            
            # Get existing conversation
            existing_conversation = self.redis_client.get(conversation_key)
            conversation = json.loads(existing_conversation) if existing_conversation else []
            
            # Add new turn
            conversation.append({
                'timestamp': datetime.now().isoformat(),
                'user_query': user_query,
                'bot_response': bot_response,
                'context': context or {}
            })
            
            # Keep only recent conversation (configurable limit)
            max_turns = settings.max_conversation_history
            if len(conversation) > max_turns:
                conversation = conversation[-max_turns:]
            
            # Store back in Redis with expiration
            self.redis_client.setex(
                conversation_key,
                timedelta(days=settings.memory_retention_days),
                json.dumps(conversation)
            )
            
            # Store user preferences and patterns
            await self._update_user_preferences(session_id, user_query, context)
            
        except Exception as e:
            logger.error(f"Error storing short-term memory: {e}")
    
    async def _store_long_term_memory(
        self,
        session_id: str,
        user_query: str,
        bot_response: str,
        context: Dict[str, Any]
    ):
        """Store significant conversations in Qdrant for long-term memory"""
        try:
            # Create embedding for the conversation
            conversation_text = f"User: {user_query}\nBot: {bot_response}"
            embedding = await self._get_embedding(conversation_text)
            
            if not embedding:
                return
            
            # Create point for Qdrant
            point = PointStruct(
                id=f"{session_id}_{datetime.now().timestamp()}",
                vector=embedding,
                payload={
                    'session_id': session_id,
                    'user_query': user_query,
                    'bot_response': bot_response,
                    'context': context or {},
                    'timestamp': datetime.now().isoformat(),
                    'significance_score': await self._calculate_significance(user_query, bot_response)
                }
            )
            
            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.memory_collection,
                points=[point]
            )
            
        except Exception as e:
            logger.error(f"Error storing long-term memory: {e}")
    
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for current session"""
        try:
            conversation_key = f"session:{session_id}:conversation"
            conversation_data = self.redis_client.get(conversation_key)
            
            if conversation_data:
                return json.loads(conversation_data)
            return []
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    async def get_relevant_memories(
        self,
        session_id: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get relevant memories from long-term storage"""
        try:
            # Get embedding for query
            query_embedding = await self._get_embedding(query)
            if not query_embedding:
                return []
            
            # Search Qdrant for similar memories
            search_results = self.qdrant_client.search(
                collection_name=self.memory_collection,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.7
            )
            
            # Format results
            memories = []
            for result in search_results:
                memories.append({
                    'user_query': result.payload.get('user_query', ''),
                    'bot_response': result.payload.get('bot_response', ''),
                    'context': result.payload.get('context', {}),
                    'timestamp': result.payload.get('timestamp', ''),
                    'relevance_score': result.score,
                    'significance_score': result.payload.get('significance_score', 0)
                })
            
            return memories
            
        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}")
            return []
    
    async def get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Get user preferences and patterns"""
        try:
            preferences_key = f"session:{session_id}:preferences"
            preferences_data = self.redis_client.get(preferences_key)
            
            if preferences_data:
                return json.loads(preferences_data)
            
            return {
                'interests': [],
                'expertise_level': 'beginner',
                'preferred_topics': [],
                'interaction_style': 'conversational',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}
    
    async def _update_user_preferences(
        self,
        session_id: str,
        user_query: str,
        context: Dict[str, Any]
    ):
        """Update user preferences based on conversation patterns"""
        try:
            preferences = await self.get_user_preferences(session_id)
            
            # Analyze query for preferences
            query_lower = user_query.lower()
            
            # Update interests based on topics mentioned
            genai_topics = {
                'llm': ['llm', 'large language model', 'gpt', 'claude', 'gemini'],
                'computer_vision': ['computer vision', 'image', 'vision', 'cv'],
                'nlp': ['nlp', 'natural language', 'text processing'],
                'multimodal': ['multimodal', 'vision-language', 'image-text'],
                'rag': ['rag', 'retrieval augmented', 'vector database'],
                'fine-tuning': ['fine-tuning', 'fine tuning', 'training'],
                'frameworks': ['langchain', 'langgraph', 'huggingface', 'transformers']
            }
            
            for topic, keywords in genai_topics.items():
                if any(keyword in query_lower for keyword in keywords):
                    if topic not in preferences.get('interests', []):
                        preferences.setdefault('interests', []).append(topic)
            
            # Update expertise level based on query complexity
            if any(term in query_lower for term in ['advanced', 'complex', 'architecture', 'implementation']):
                preferences['expertise_level'] = 'advanced'
            elif any(term in query_lower for term in ['beginner', 'basic', 'simple', 'introduction']):
                preferences['expertise_level'] = 'beginner'
            else:
                preferences['expertise_level'] = 'intermediate'
            
            # Update interaction style
            if any(term in query_lower for term in ['detailed', 'comprehensive', 'thorough']):
                preferences['interaction_style'] = 'detailed'
            elif any(term in query_lower for term in ['brief', 'quick', 'summary']):
                preferences['interaction_style'] = 'concise'
            
            preferences['last_updated'] = datetime.now().isoformat()
            
            # Store updated preferences
            preferences_key = f"session:{session_id}:preferences"
            self.redis_client.setex(
                preferences_key,
                timedelta(days=settings.memory_retention_days),
                json.dumps(preferences)
            )
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
    
    async def _is_significant_conversation(self, user_query: str, bot_response: str) -> bool:
        """Determine if conversation is significant enough for long-term storage"""
        try:
            # Simple heuristics for significance
            significance_indicators = [
                len(user_query) > 50,  # Substantial query
                len(bot_response) > 200,  # Substantial response
                any(term in user_query.lower() for term in [
                    'how to', 'tutorial', 'guide', 'best practice',
                    'recommendation', 'comparison', 'analysis'
                ]),
                any(term in bot_response.lower() for term in [
                    'according to', 'research shows', 'studies indicate',
                    'best practice', 'recommendation'
                ])
            ]
            
            return sum(significance_indicators) >= 2
            
        except Exception as e:
            logger.error(f"Error determining conversation significance: {e}")
            return False
    
    async def _calculate_significance(self, user_query: str, bot_response: str) -> float:
        """Calculate significance score for conversation"""
        try:
            score = 0.0
            
            # Length factors
            if len(user_query) > 50:
                score += 0.2
            if len(bot_response) > 200:
                score += 0.2
            
            # Content factors
            if any(term in user_query.lower() for term in ['how to', 'tutorial', 'guide']):
                score += 0.3
            if any(term in bot_response.lower() for term in ['according to', 'research shows']):
                score += 0.3
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating significance: {e}")
            return 0.5
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    async def get_memory_context(self, session_id: str, current_query: str) -> Dict[str, Any]:
        """Get comprehensive memory context for current query"""
        try:
            # Get conversation history
            conversation_history = await self.get_conversation_history(session_id)
            
            # Get relevant long-term memories
            relevant_memories = await self.get_relevant_memories(session_id, current_query)
            
            # Get user preferences
            user_preferences = await self.get_user_preferences(session_id)
            
            return {
                'conversation_history': conversation_history[-3:],  # Last 3 turns
                'relevant_memories': relevant_memories,
                'user_preferences': user_preferences,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting memory context: {e}")
            return {
                'conversation_history': [],
                'relevant_memories': [],
                'user_preferences': {},
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
    
    async def cleanup_old_memories(self, days_old: int = 30):
        """Clean up old memories from Redis"""
        try:
            # This would implement cleanup logic for old Redis keys
            # For now, Redis TTL handles this automatically
            logger.info(f"Memory cleanup completed for keys older than {days_old} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")
