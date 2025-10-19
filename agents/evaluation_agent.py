"""
Evaluation Agent for validating responses and measuring quality
"""
import asyncio
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from config import settings

# Import evaluation libraries (these would be installed via requirements.txt)
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("Ragas not available. Using custom evaluation metrics.")

try:
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logging.warning("DeepEval not available. Using custom evaluation metrics.")

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Structure for evaluation results"""
    query: str
    response: str
    sources: List[Dict[str, Any]]
    hallucination_score: float
    relevance_score: float
    context_precision: float
    context_recall: float
    overall_score: float
    timestamp: str
    evaluation_metadata: Dict[str, Any]


class EvaluationAgent:
    """Handles response evaluation and quality metrics"""
    
    def __init__(self):
        self.evaluation_results = []
        self.evaluation_threshold = settings.evaluation_threshold
        
    async def evaluate_response(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]] = None,
        context: List[str] = None
    ) -> EvaluationResult:
        """Evaluate a response using multiple metrics"""
        try:
            # Initialize scores
            hallucination_score = 0.0
            relevance_score = 0.0
            context_precision = 0.0
            context_recall = 0.0
            
            # Use available evaluation libraries
            if RAGAS_AVAILABLE:
                ragas_scores = await self._evaluate_with_ragas(query, response, context or [])
                hallucination_score = ragas_scores.get('faithfulness', 0.0)
                relevance_score = ragas_scores.get('answer_relevancy', 0.0)
                context_precision = ragas_scores.get('context_precision', 0.0)
                context_recall = ragas_scores.get('context_recall', 0.0)
            elif DEEPEVAL_AVAILABLE:
                deepeval_scores = await self._evaluate_with_deepeval(query, response, context or [])
                hallucination_score = deepeval_scores.get('hallucination', 0.0)
                relevance_score = deepeval_scores.get('relevance', 0.0)
            else:
                # Use custom evaluation methods
                custom_scores = await self._evaluate_with_custom_methods(query, response, sources or [])
                hallucination_score = custom_scores.get('hallucination', 0.0)
                relevance_score = custom_scores.get('relevance', 0.0)
                context_precision = custom_scores.get('context_precision', 0.0)
                context_recall = custom_scores.get('context_recall', 0.0)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                hallucination_score, relevance_score, context_precision, context_recall
            )
            
            # Create evaluation result
            result = EvaluationResult(
                query=query,
                response=response,
                sources=sources or [],
                hallucination_score=hallucination_score,
                relevance_score=relevance_score,
                context_precision=context_precision,
                context_recall=context_recall,
                overall_score=overall_score,
                timestamp=datetime.now().isoformat(),
                evaluation_metadata={
                    'evaluation_method': 'ragas' if RAGAS_AVAILABLE else 'deepeval' if DEEPEVAL_AVAILABLE else 'custom',
                    'context_length': len(context) if context else 0,
                    'sources_count': len(sources) if sources else 0
                }
            )
            
            # Store result
            self.evaluation_results.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            # Return default evaluation result
            return EvaluationResult(
                query=query,
                response=response,
                sources=sources or [],
                hallucination_score=0.5,
                relevance_score=0.5,
                context_precision=0.5,
                context_recall=0.5,
                overall_score=0.5,
                timestamp=datetime.now().isoformat(),
                evaluation_metadata={'error': str(e)}
            )
    
    async def _evaluate_with_ragas(
        self,
        query: str,
        response: str,
        context: List[str]
    ) -> Dict[str, float]:
        """Evaluate using Ragas metrics"""
        try:
            # Prepare data for Ragas evaluation
            dataset = [{
                'question': query,
                'answer': response,
                'contexts': context,
                'ground_truth': ""  # Would need ground truth for full evaluation
            }]
            
            # Define metrics
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
            
            # Run evaluation
            result = evaluate(dataset, metrics=metrics)
            
            return {
                'faithfulness': result.get('faithfulness', 0.0),
                'answer_relevancy': result.get('answer_relevancy', 0.0),
                'context_precision': result.get('context_precision', 0.0),
                'context_recall': result.get('context_recall', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error with Ragas evaluation: {e}")
            return {}
    
    async def _evaluate_with_deepeval(
        self,
        query: str,
        response: str,
        context: List[str]
    ) -> Dict[str, float]:
        """Evaluate using DeepEval metrics"""
        try:
            # This would implement DeepEval evaluation
            # For now, return placeholder scores
            return {
                'hallucination': 0.8,  # Placeholder
                'relevance': 0.7       # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Error with DeepEval evaluation: {e}")
            return {}
    
    async def _evaluate_with_custom_methods(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Custom evaluation methods when external libraries aren't available"""
        try:
            # Hallucination detection (simple heuristics)
            hallucination_score = await self._detect_hallucination(response, sources)
            
            # Relevance scoring
            relevance_score = await self._calculate_relevance(query, response)
            
            # Context precision and recall
            context_precision = await self._calculate_context_precision(response, sources)
            context_recall = await self._calculate_context_recall(query, sources)
            
            return {
                'hallucination': hallucination_score,
                'relevance': relevance_score,
                'context_precision': context_precision,
                'context_recall': context_recall
            }
            
        except Exception as e:
            logger.error(f"Error with custom evaluation: {e}")
            return {}
    
    async def _detect_hallucination(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """Simple hallucination detection"""
        try:
            if not sources:
                return 0.3  # Lower score if no sources
            
            # Check for unsupported claims
            unsupported_indicators = [
                'definitely', 'certainly', 'always', 'never', 'all', 'none',
                'proven', 'guaranteed', 'impossible'
            ]
            
            response_lower = response.lower()
            unsupported_count = sum(1 for indicator in unsupported_indicators if indicator in response_lower)
            
            # Check for source citations
            citation_indicators = ['according to', 'research shows', 'studies indicate', 'source:', 'reference:']
            citation_count = sum(1 for indicator in citation_indicators if indicator in response_lower)
            
            # Calculate score (higher is better, meaning less hallucination)
            base_score = 0.7
            if citation_count > 0:
                base_score += 0.2
            if unsupported_count > 2:
                base_score -= 0.3
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.error(f"Error detecting hallucination: {e}")
            return 0.5
    
    async def _calculate_relevance(self, query: str, response: str) -> float:
        """Calculate relevance between query and response"""
        try:
            # Simple keyword overlap scoring
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            if not query_words:
                return 0.0
            
            overlap = len(query_words.intersection(response_words))
            relevance_score = overlap / len(query_words)
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.5
    
    async def _calculate_context_precision(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate how precise the context usage is"""
        try:
            if not sources:
                return 0.0
            
            # Check if response references source content
            response_lower = response.lower()
            source_content = " ".join([source.get('content', '') for source in sources]).lower()
            
            # Simple overlap calculation
            response_words = set(response_lower.split())
            source_words = set(source_content.split())
            
            if not source_words:
                return 0.0
            
            overlap = len(response_words.intersection(source_words))
            precision = overlap / len(response_words) if response_words else 0.0
            
            return min(1.0, precision)
            
        except Exception as e:
            logger.error(f"Error calculating context precision: {e}")
            return 0.5
    
    async def _calculate_context_recall(
        self,
        query: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate how well sources cover the query"""
        try:
            if not sources:
                return 0.0
            
            query_words = set(query.lower().split())
            all_source_content = " ".join([source.get('content', '') for source in sources]).lower()
            source_words = set(all_source_content.split())
            
            if not query_words:
                return 0.0
            
            overlap = len(query_words.intersection(source_words))
            recall = overlap / len(query_words)
            
            return min(1.0, recall)
            
        except Exception as e:
            logger.error(f"Error calculating context recall: {e}")
            return 0.5
    
    def _calculate_overall_score(
        self,
        hallucination_score: float,
        relevance_score: float,
        context_precision: float,
        context_recall: float
    ) -> float:
        """Calculate weighted overall score"""
        # Weighted average with emphasis on hallucination and relevance
        weights = {
            'hallucination': 0.4,
            'relevance': 0.3,
            'context_precision': 0.15,
            'context_recall': 0.15
        }
        
        overall_score = (
            hallucination_score * weights['hallucination'] +
            relevance_score * weights['relevance'] +
            context_precision * weights['context_precision'] +
            context_recall * weights['context_recall']
        )
        
        return overall_score
    
    async def batch_evaluate(
        self,
        queries_responses: List[Tuple[str, str, List[Dict[str, Any]]]]
    ) -> List[EvaluationResult]:
        """Evaluate multiple query-response pairs"""
        try:
            results = []
            
            for query, response, sources in queries_responses:
                result = await self.evaluate_response(query, response, sources)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch evaluation: {e}")
            return []
    
    def export_evaluation_results(self, filename: str = None) -> str:
        """Export evaluation results to CSV"""
        try:
            if not filename:
                filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Convert results to DataFrame
            data = []
            for result in self.evaluation_results:
                data.append({
                    'query': result.query,
                    'response': result.response,
                    'hallucination_score': result.hallucination_score,
                    'relevance_score': result.relevance_score,
                    'context_precision': result.context_precision,
                    'context_recall': result.context_recall,
                    'overall_score': result.overall_score,
                    'timestamp': result.timestamp,
                    'sources_count': len(result.sources),
                    'evaluation_method': result.evaluation_metadata.get('evaluation_method', 'unknown')
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            logger.info(f"Exported {len(data)} evaluation results to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting evaluation results: {e}")
            return ""
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of evaluations"""
        try:
            if not self.evaluation_results:
                return {}
            
            scores = {
                'hallucination': [r.hallucination_score for r in self.evaluation_results],
                'relevance': [r.relevance_score for r in self.evaluation_results],
                'context_precision': [r.context_precision for r in self.evaluation_results],
                'context_recall': [r.context_recall for r in self.evaluation_results],
                'overall': [r.overall_score for r in self.evaluation_results]
            }
            
            summary = {}
            for metric, score_list in scores.items():
                if score_list:
                    summary[metric] = {
                        'mean': sum(score_list) / len(score_list),
                        'min': min(score_list),
                        'max': max(score_list),
                        'count': len(score_list)
                    }
            
            # Calculate pass rate (above threshold)
            pass_count = sum(1 for r in self.evaluation_results if r.overall_score >= self.evaluation_threshold)
            summary['pass_rate'] = pass_count / len(self.evaluation_results) if self.evaluation_results else 0
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting evaluation summary: {e}")
            return {}
    
    async def validate_response_quality(self, result: EvaluationResult) -> bool:
        """Validate if response meets quality threshold"""
        return result.overall_score >= self.evaluation_threshold
