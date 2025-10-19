"""
Retrieval Agent for querying Qdrant vector database
"""
import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from config import settings

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """Handles vector database operations and document retrieval"""
    
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=settings.google_api_key,
            model=settings.embedding_model
        )
        self.collection_name = "genai_knowledge"
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            if self.collection_name not in [c.name for c in collections.collections]:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=3072,  # Gemini gemini-embedding-001 embedding size
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector database"""
        try:
            points = []
            
            for doc in documents:
                # Split document into chunks
                chunks = self.text_splitter.split_text(doc.get('content', ''))
                
                for i, chunk in enumerate(chunks):
                    # Generate embedding
                    embedding = await self._get_embedding(chunk)
                    
                    # Create point with UUID-like ID
                    import uuid
                    point_id = str(uuid.uuid4())
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            'content': chunk,
                            'source': doc.get('source', 'unknown'),
                            'title': doc.get('title', ''),
                            'url': doc.get('url', ''),
                            'metadata': doc.get('metadata', {}),
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'original_doc_id': doc.get('id', 'unknown')
                        }
                    )
                    points.append(point)
            
            # Insert points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"Added {len(points)} document chunks to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    async def search_similar_documents(
        self, 
        query: str, 
        limit: int = 5,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = await self._get_embedding(query)
            if not query_embedding:
                return []
            
            # Build filter conditions
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Search vector database
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    'content': result.payload.get('content', ''),
                    'source': result.payload.get('source', ''),
                    'title': result.payload.get('title', ''),
                    'url': result.payload.get('url', ''),
                    'metadata': result.payload.get('metadata', {}),
                    'score': result.score,
                    'chunk_index': result.payload.get('chunk_index', 0),
                    'total_chunks': result.payload.get('total_chunks', 1)
                })
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def hybrid_search(
        self, 
        query: str, 
        limit: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and keyword search"""
        try:
            # Vector search
            vector_results = await self.search_similar_documents(query, limit * 2)
            
            # Keyword search (simple implementation)
            keyword_results = await self._keyword_search(query, limit * 2)
            
            # Combine and rank results
            combined_results = self._combine_search_results(
                vector_results, keyword_results, vector_weight, keyword_weight
            )
            
            return combined_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    async def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Simple keyword search implementation"""
        try:
            # This is a simplified keyword search
            # In production, you might want to use a more sophisticated approach
            keywords = query.lower().split()
            
            # Get all points and filter by keywords
            all_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000  # Adjust based on your data size
            )[0]
            
            results = []
            for point in all_points:
                content = point.payload.get('content', '').lower()
                score = sum(1 for keyword in keywords if keyword in content) / len(keywords)
                
                if score > 0:
                    results.append({
                        'content': point.payload.get('content', ''),
                        'source': point.payload.get('source', ''),
                        'title': point.payload.get('title', ''),
                        'url': point.payload.get('url', ''),
                        'metadata': point.payload.get('metadata', {}),
                        'score': score,
                        'chunk_index': point.payload.get('chunk_index', 0),
                        'total_chunks': point.payload.get('total_chunks', 1)
                    })
            
            # Sort by score and return top results
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _combine_search_results(
        self, 
        vector_results: List[Dict[str, Any]], 
        keyword_results: List[Dict[str, Any]],
        vector_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine and rank search results"""
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Add vector search results
        for result in vector_results:
            doc_id = f"{result['source']}_{result['chunk_index']}"
            combined_scores[doc_id] = {
                'result': result,
                'vector_score': result['score'],
                'keyword_score': 0
            }
        
        # Add keyword search results
        for result in keyword_results:
            doc_id = f"{result['source']}_{result['chunk_index']}"
            if doc_id in combined_scores:
                combined_scores[doc_id]['keyword_score'] = result['score']
            else:
                combined_scores[doc_id] = {
                    'result': result,
                    'vector_score': 0,
                    'keyword_score': result['score']
                }
        
        # Calculate combined scores and sort
        final_results = []
        for doc_id, scores in combined_scores.items():
            combined_score = (
                scores['vector_score'] * vector_weight + 
                scores['keyword_score'] * keyword_weight
            )
            result = scores['result'].copy()
            result['combined_score'] = combined_score
            final_results.append(result)
        
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return final_results
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'name': collection_info.config.params.vectors.size,
                'vectors_count': collection_info.vectors_count,
                'indexed_vectors_count': collection_info.indexed_vectors_count,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
