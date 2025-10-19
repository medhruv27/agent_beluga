"""
Cost optimization and prompt compression utilities
"""
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Track token usage for cost optimization"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    model: str
    timestamp: str


class PromptCompressor:
    """Handles prompt compression to reduce token usage"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        # Use cheaper model for compression
        self.compression_llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=settings.google_api_key,
            temperature=0.1,
            max_output_tokens=200
        )
    
    async def compress_conversation_history(
        self,
        conversation_history: List[Dict[str, Any]],
        max_tokens: int = 1000
    ) -> str:
        """Compress conversation history to reduce token usage"""
        try:
            if not conversation_history:
                return ""
            
            # Extract key information from conversation
            key_points = []
            for turn in conversation_history[-5:]:  # Last 5 turns
                if turn.get('type') == 'user':
                    key_points.append(f"User asked: {turn.get('content', '')[:100]}")
                elif turn.get('type') == 'bot':
                    # Extract key insights from bot responses
                    response = turn.get('content', '')
                    if len(response) > 100:
                        # Summarize long responses
                        summary = await self._summarize_text(response)
                        key_points.append(f"Bot explained: {summary}")
                    else:
                        key_points.append(f"Bot said: {response}")
            
            # Compress key points
            compressed_history = "\n".join(key_points)
            
            # Further compress if still too long
            if len(compressed_history) > max_tokens:
                compressed_history = await self._compress_text(compressed_history, max_tokens)
            
            return compressed_history
            
        except Exception as e:
            logger.error(f"Error compressing conversation history: {e}")
            return ""
    
    async def _summarize_text(self, text: str) -> str:
        """Summarize text using LLM"""
        try:
            prompt = f"""
            Summarize the following text in 1-2 sentences, focusing on key points:
            
            {text[:500]}
            
            Summary:
            """
            
            response = await self.compression_llm.agenerate([prompt])
            return response.generations[0][0].text.strip()
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text[:100] + "..."
    
    async def _compress_text(self, text: str, max_tokens: int) -> str:
        """Compress text to fit within token limit"""
        try:
            # Simple compression by truncating and adding ellipsis
            if len(text) <= max_tokens:
                return text
            
            # Try to find a good breaking point
            truncated = text[:max_tokens-50]
            last_period = truncated.rfind('.')
            if last_period > max_tokens * 0.7:  # If we can keep 70% of content
                return truncated[:last_period+1] + "..."
            else:
                return truncated + "..."
                
        except Exception as e:
            logger.error(f"Error compressing text: {e}")
            return text[:max_tokens] + "..."
    
    async def compress_retrieved_documents(
        self,
        documents: List[Dict[str, Any]],
        max_documents: int = 3
    ) -> List[Dict[str, Any]]:
        """Compress retrieved documents to reduce context size"""
        try:
            if not documents:
                return []
            
            # Sort by relevance score and take top documents
            sorted_docs = sorted(
                documents,
                key=lambda x: x.get('score', 0) or x.get('combined_score', 0),
                reverse=True
            )
            
            compressed_docs = []
            for doc in sorted_docs[:max_documents]:
                # Compress document content
                content = doc.get('content', '')
                if len(content) > 500:
                    compressed_content = await self._summarize_text(content)
                else:
                    compressed_content = content
                
                compressed_docs.append({
                    'content': compressed_content,
                    'source': doc.get('source', ''),
                    'title': doc.get('title', ''),
                    'url': doc.get('url', ''),
                    'score': doc.get('score', 0) or doc.get('combined_score', 0)
                })
            
            return compressed_docs
            
        except Exception as e:
            logger.error(f"Error compressing retrieved documents: {e}")
            return documents[:max_documents]


class CostOptimizer:
    """Handles cost optimization strategies"""
    
    def __init__(self):
        self.token_usage_history: List[TokenUsage] = []
        self.cost_thresholds = {
            'daily': 10.0,  # $10 per day
            'monthly': 200.0,  # $200 per month
            'per_query': 0.50  # $0.50 per query
        }
        
        # Model costs (per 1K tokens) - Gemini pricing
        self.model_costs = {
            'gemini-pro': {'input': 0.0005, 'output': 0.0015},
            'gemini-pro-vision': {'input': 0.0005, 'output': 0.0015},
            'text-embedding-004': {'input': 0.000025, 'output': 0.0}
        }
    
    def calculate_token_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ) -> float:
        """Calculate cost for token usage"""
        try:
            if model not in self.model_costs:
                model = 'gpt-3.5-turbo'  # Default to cheaper model
            
            costs = self.model_costs[model]
            input_cost = (prompt_tokens / 1000) * costs['input']
            output_cost = (completion_tokens / 1000) * costs['output']
            
            return input_cost + output_cost
            
        except Exception as e:
            logger.error(f"Error calculating token cost: {e}")
            return 0.0
    
    def record_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ):
        """Record token usage for tracking"""
        try:
            total_tokens = prompt_tokens + completion_tokens
            cost = self.calculate_token_cost(prompt_tokens, completion_tokens, model)
            
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost,
                model=model,
                timestamp=datetime.now().isoformat()
            )
            
            self.token_usage_history.append(usage)
            
            # Keep only last 1000 records
            if len(self.token_usage_history) > 1000:
                self.token_usage_history = self.token_usage_history[-1000:]
            
        except Exception as e:
            logger.error(f"Error recording token usage: {e}")
    
    def get_daily_cost(self) -> float:
        """Get total cost for today"""
        try:
            today = datetime.now().date()
            daily_usage = [
                usage for usage in self.token_usage_history
                if datetime.fromisoformat(usage.timestamp).date() == today
            ]
            return sum(usage.cost for usage in daily_usage)
        except Exception as e:
            logger.error(f"Error calculating daily cost: {e}")
            return 0.0
    
    def get_monthly_cost(self) -> float:
        """Get total cost for this month"""
        try:
            now = datetime.now()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            monthly_usage = [
                usage for usage in self.token_usage_history
                if datetime.fromisoformat(usage.timestamp) >= month_start
            ]
            return sum(usage.cost for usage in monthly_usage)
        except Exception as e:
            logger.error(f"Error calculating monthly cost: {e}")
            return 0.0
    
    def should_use_cheaper_model(self, query_complexity: str = "medium") -> str:
        """Determine which model to use based on cost and complexity"""
        try:
            daily_cost = self.get_daily_cost()
            
            # Use cheaper model if approaching daily limit
            if daily_cost > self.cost_thresholds['daily'] * 0.8:
                return 'gemini-pro'
            
            # Use cheaper model for simple queries
            if query_complexity == "simple":
                return 'gemini-pro'
            
            # Use more powerful model for complex queries if within budget
            if query_complexity == "complex" and daily_cost < self.cost_thresholds['daily'] * 0.5:
                return 'gemini-pro'
            
            # Default to balanced model
            return 'gemini-pro'
            
        except Exception as e:
            logger.error(f"Error determining model: {e}")
            return 'gemini-pro'
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary"""
        try:
            daily_cost = self.get_daily_cost()
            monthly_cost = self.get_monthly_cost()
            total_queries = len(self.token_usage_history)
            
            # Calculate average cost per query
            avg_cost_per_query = monthly_cost / total_queries if total_queries > 0 else 0
            
            # Model usage breakdown
            model_usage = {}
            for usage in self.token_usage_history:
                model = usage.model
                if model not in model_usage:
                    model_usage[model] = {'count': 0, 'cost': 0.0, 'tokens': 0}
                model_usage[model]['count'] += 1
                model_usage[model]['cost'] += usage.cost
                model_usage[model]['tokens'] += usage.total_tokens
            
            return {
                'daily_cost': daily_cost,
                'monthly_cost': monthly_cost,
                'total_queries': total_queries,
                'avg_cost_per_query': avg_cost_per_query,
                'model_usage': model_usage,
                'cost_thresholds': self.cost_thresholds,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating cost summary: {e}")
            return {}


class QueryComplexityAnalyzer:
    """Analyzes query complexity to optimize model selection"""
    
    def __init__(self):
        self.complexity_indicators = {
            'simple': [
                'what is', 'define', 'explain briefly', 'simple', 'basic',
                'quick', 'short', 'overview'
            ],
            'medium': [
                'how to', 'compare', 'difference between', 'pros and cons',
                'tutorial', 'guide', 'best practice'
            ],
            'complex': [
                'analyze', 'detailed analysis', 'comprehensive', 'architecture',
                'implementation', 'advanced', 'research', 'technical deep dive',
                'performance comparison', 'benchmark'
            ]
        }
    
    def analyze_complexity(self, query: str) -> str:
        """Analyze query complexity"""
        try:
            query_lower = query.lower()
            
            # Check for complex indicators
            for indicator in self.complexity_indicators['complex']:
                if indicator in query_lower:
                    return 'complex'
            
            # Check for medium indicators
            for indicator in self.complexity_indicators['medium']:
                if indicator in query_lower:
                    return 'medium'
            
            # Check for simple indicators
            for indicator in self.complexity_indicators['simple']:
                if indicator in query_lower:
                    return 'simple'
            
            # Default based on query length and structure
            if len(query.split()) > 20:
                return 'complex'
            elif len(query.split()) > 10:
                return 'medium'
            else:
                return 'simple'
                
        except Exception as e:
            logger.error(f"Error analyzing query complexity: {e}")
            return 'medium'


class OptimizationManager:
    """Main optimization manager that coordinates all optimization strategies"""
    
    def __init__(self):
        self.prompt_compressor = PromptCompressor()
        self.cost_optimizer = CostOptimizer()
        self.complexity_analyzer = QueryComplexityAnalyzer()
    
    async def optimize_query_processing(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        retrieved_documents: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """Optimize query processing for cost and efficiency"""
        try:
            # Analyze query complexity
            complexity = self.complexity_analyzer.analyze_complexity(query)
            
            # Determine optimal model
            optimal_model = self.cost_optimizer.should_use_cheaper_model(complexity)
            
            # Compress conversation history
            compressed_history = await self.prompt_compressor.compress_conversation_history(
                conversation_history
            )
            
            # Compress retrieved documents
            compressed_docs = await self.prompt_compressor.compress_retrieved_documents(
                retrieved_documents
            )
            
            return compressed_history, compressed_docs, optimal_model
            
        except Exception as e:
            logger.error(f"Error optimizing query processing: {e}")
            return "", retrieved_documents, 'gpt-3.5-turbo'
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        try:
            cost_summary = self.cost_optimizer.get_cost_summary()
            
            return {
                'cost_optimization': cost_summary,
                'optimization_strategies': {
                    'prompt_compression': 'Active',
                    'model_selection': 'Dynamic based on complexity',
                    'document_compression': 'Active',
                    'conversation_summarization': 'Active'
                },
                'recommendations': self._generate_recommendations(cost_summary),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating optimization report: {e}")
            return {}
    
    def _generate_recommendations(self, cost_summary: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        try:
            daily_cost = cost_summary.get('daily_cost', 0)
            monthly_cost = cost_summary.get('monthly_cost', 0)
            
            if daily_cost > self.cost_optimizer.cost_thresholds['daily'] * 0.8:
                recommendations.append("Consider using cheaper models for simple queries")
            
            if monthly_cost > self.cost_optimizer.cost_thresholds['monthly'] * 0.8:
                recommendations.append("Monthly cost approaching limit - consider reducing query frequency")
            
            avg_cost = cost_summary.get('avg_cost_per_query', 0)
            if avg_cost > self.cost_optimizer.cost_thresholds['per_query']:
                recommendations.append("Average cost per query is high - optimize prompt compression")
            
            if not recommendations:
                recommendations.append("Cost optimization is working well - no immediate recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
