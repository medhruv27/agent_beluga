"""
Evaluation metrics and CSV reporting for the GenAI chatbot
"""
import asyncio
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import os

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics structure"""
    # Basic metrics
    query: str
    response: str
    timestamp: str
    session_id: str
    
    # Quality metrics
    hallucination_rate: float
    contextual_relevance: float
    answer_accuracy: float
    response_completeness: float
    
    # Agent performance metrics
    retrieval_agent_score: float
    webscraper_agent_score: float
    reasoning_agent_score: float
    memory_agent_score: float
    evaluation_agent_score: float
    
    # Human-like behavior metrics
    persona_consistency: float
    engagement_level: float
    clarification_questions_asked: int
    off_topic_redirects: int
    
    # Memory utilization metrics
    short_term_memory_used: bool
    long_term_memory_used: bool
    user_preferences_utilized: bool
    conversation_context_used: bool
    
    # Custom evaluation matrix
    domain_specificity: float
    technical_accuracy: float
    practical_utility: float
    response_continuity: float
    
    # Overall scores
    overall_quality_score: float
    human_like_behavior_score: float
    multi_agent_architecture_score: float
    memory_utilization_score: float
    
    # Metadata
    response_time_seconds: float
    token_usage: int
    cost_estimate: float
    evaluation_method: str


class EvaluationMetricsCollector:
    """Collects and manages evaluation metrics"""
    
    def __init__(self):
        self.metrics_history: List[EvaluationMetrics] = []
        self.evaluation_thresholds = {
            'hallucination_rate': 0.2,  # Lower is better
            'contextual_relevance': 0.8,  # Higher is better
            'overall_quality_score': 0.8,  # Higher is better
            'human_like_behavior_score': 0.7,  # Higher is better
            'multi_agent_architecture_score': 0.8,  # Higher is better
            'memory_utilization_score': 0.7  # Higher is better
        }
    
    async def collect_comprehensive_metrics(
        self,
        query: str,
        response: str,
        session_id: str,
        agent_performance: Dict[str, Any],
        response_metadata: Dict[str, Any]
    ) -> EvaluationMetrics:
        """Collect comprehensive evaluation metrics"""
        try:
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(query, response, response_metadata)
            
            # Calculate agent performance metrics
            agent_metrics = await self._calculate_agent_metrics(agent_performance)
            
            # Calculate human-like behavior metrics
            behavior_metrics = await self._calculate_behavior_metrics(query, response, response_metadata)
            
            # Calculate memory utilization metrics
            memory_metrics = await self._calculate_memory_metrics(response_metadata)
            
            # Calculate custom evaluation matrix
            custom_metrics = await self._calculate_custom_metrics(query, response, response_metadata)
            
            # Calculate overall scores
            overall_scores = await self._calculate_overall_scores(
                quality_metrics, agent_metrics, behavior_metrics, memory_metrics, custom_metrics
            )
            
            # Create comprehensive metrics object
            metrics = EvaluationMetrics(
                # Basic metrics
                query=query,
                response=response,
                timestamp=datetime.now().isoformat(),
                session_id=session_id,
                
                # Quality metrics
                hallucination_rate=quality_metrics['hallucination_rate'],
                contextual_relevance=quality_metrics['contextual_relevance'],
                answer_accuracy=quality_metrics['answer_accuracy'],
                response_completeness=quality_metrics['response_completeness'],
                
                # Agent performance metrics
                retrieval_agent_score=agent_metrics['retrieval_agent_score'],
                webscraper_agent_score=agent_metrics['webscraper_agent_score'],
                reasoning_agent_score=agent_metrics['reasoning_agent_score'],
                memory_agent_score=agent_metrics['memory_agent_score'],
                evaluation_agent_score=agent_metrics['evaluation_agent_score'],
                
                # Human-like behavior metrics
                persona_consistency=behavior_metrics['persona_consistency'],
                engagement_level=behavior_metrics['engagement_level'],
                clarification_questions_asked=behavior_metrics['clarification_questions_asked'],
                off_topic_redirects=behavior_metrics['off_topic_redirects'],
                
                # Memory utilization metrics
                short_term_memory_used=memory_metrics['short_term_memory_used'],
                long_term_memory_used=memory_metrics['long_term_memory_used'],
                user_preferences_utilized=memory_metrics['user_preferences_utilized'],
                conversation_context_used=memory_metrics['conversation_context_used'],
                
                # Custom evaluation matrix
                domain_specificity=custom_metrics['domain_specificity'],
                technical_accuracy=custom_metrics['technical_accuracy'],
                practical_utility=custom_metrics['practical_utility'],
                response_continuity=custom_metrics['response_continuity'],
                
                # Overall scores
                overall_quality_score=overall_scores['overall_quality_score'],
                human_like_behavior_score=overall_scores['human_like_behavior_score'],
                multi_agent_architecture_score=overall_scores['multi_agent_architecture_score'],
                memory_utilization_score=overall_scores['memory_utilization_score'],
                
                # Metadata
                response_time_seconds=response_metadata.get('response_time', 0.0),
                token_usage=response_metadata.get('token_usage', 0),
                cost_estimate=response_metadata.get('cost_estimate', 0.0),
                evaluation_method=response_metadata.get('evaluation_method', 'comprehensive')
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting comprehensive metrics: {e}")
            # Return default metrics
            return self._create_default_metrics(query, response, session_id)
    
    async def _calculate_quality_metrics(
        self,
        query: str,
        response: str,
        response_metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate quality-related metrics"""
        try:
            # Hallucination rate (simplified detection)
            hallucination_rate = await self._detect_hallucination_rate(response, response_metadata)
            
            # Contextual relevance
            contextual_relevance = await self._calculate_contextual_relevance(query, response)
            
            # Answer accuracy (based on sources and confidence)
            answer_accuracy = response_metadata.get('confidence', 0.5)
            
            # Response completeness
            response_completeness = await self._calculate_response_completeness(query, response)
            
            return {
                'hallucination_rate': hallucination_rate,
                'contextual_relevance': contextual_relevance,
                'answer_accuracy': answer_accuracy,
                'response_completeness': response_completeness
            }
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {
                'hallucination_rate': 0.5,
                'contextual_relevance': 0.5,
                'answer_accuracy': 0.5,
                'response_completeness': 0.5
            }
    
    async def _calculate_agent_metrics(self, agent_performance: Dict[str, Any]) -> Dict[str, float]:
        """Calculate agent performance metrics"""
        try:
            return {
                'retrieval_agent_score': agent_performance.get('retrieval_score', 0.8),
                'webscraper_agent_score': agent_performance.get('webscraper_score', 0.7),
                'reasoning_agent_score': agent_performance.get('reasoning_score', 0.8),
                'memory_agent_score': agent_performance.get('memory_score', 0.7),
                'evaluation_agent_score': agent_performance.get('evaluation_score', 0.8)
            }
        except Exception as e:
            logger.error(f"Error calculating agent metrics: {e}")
            return {
                'retrieval_agent_score': 0.5,
                'webscraper_agent_score': 0.5,
                'reasoning_agent_score': 0.5,
                'memory_agent_score': 0.5,
                'evaluation_agent_score': 0.5
            }
    
    async def _calculate_behavior_metrics(
        self,
        query: str,
        response: str,
        response_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate human-like behavior metrics"""
        try:
            # Persona consistency
            persona_consistency = await self._check_persona_consistency(response)
            
            # Engagement level
            engagement_level = await self._calculate_engagement_level(response)
            
            # Clarification questions asked
            clarification_questions_asked = 1 if response_metadata.get('requires_clarification', False) else 0
            
            # Off-topic redirects
            off_topic_redirects = 1 if 'redirect' in response_metadata.get('type', '') else 0
            
            return {
                'persona_consistency': persona_consistency,
                'engagement_level': engagement_level,
                'clarification_questions_asked': clarification_questions_asked,
                'off_topic_redirects': off_topic_redirects
            }
            
        except Exception as e:
            logger.error(f"Error calculating behavior metrics: {e}")
            return {
                'persona_consistency': 0.5,
                'engagement_level': 0.5,
                'clarification_questions_asked': 0,
                'off_topic_redirects': 0
            }
    
    async def _calculate_memory_metrics(self, response_metadata: Dict[str, Any]) -> Dict[str, bool]:
        """Calculate memory utilization metrics"""
        try:
            memory_context = response_metadata.get('memory_context', {})
            
            return {
                'short_term_memory_used': len(memory_context.get('conversation_history', [])) > 0,
                'long_term_memory_used': len(memory_context.get('relevant_memories', [])) > 0,
                'user_preferences_utilized': bool(memory_context.get('user_preferences', {})),
                'conversation_context_used': bool(memory_context.get('conversation_history', []))
            }
            
        except Exception as e:
            logger.error(f"Error calculating memory metrics: {e}")
            return {
                'short_term_memory_used': False,
                'long_term_memory_used': False,
                'user_preferences_utilized': False,
                'conversation_context_used': False
            }
    
    async def _calculate_custom_metrics(
        self,
        query: str,
        response: str,
        response_metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate custom evaluation matrix metrics"""
        try:
            # Domain specificity (how well it stays on GenAI topics)
            domain_specificity = await self._calculate_domain_specificity(query, response)
            
            # Technical accuracy
            technical_accuracy = await self._calculate_technical_accuracy(response, response_metadata)
            
            # Practical utility
            practical_utility = await self._calculate_practical_utility(response)
            
            # Response continuity
            response_continuity = await self._calculate_response_continuity(response_metadata)
            
            return {
                'domain_specificity': domain_specificity,
                'technical_accuracy': technical_accuracy,
                'practical_utility': practical_utility,
                'response_continuity': response_continuity
            }
            
        except Exception as e:
            logger.error(f"Error calculating custom metrics: {e}")
            return {
                'domain_specificity': 0.5,
                'technical_accuracy': 0.5,
                'practical_utility': 0.5,
                'response_continuity': 0.5
            }
    
    async def _calculate_overall_scores(
        self,
        quality_metrics: Dict[str, float],
        agent_metrics: Dict[str, float],
        behavior_metrics: Dict[str, Any],
        memory_metrics: Dict[str, bool],
        custom_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate overall scores"""
        try:
            # Overall quality score (weighted average of quality metrics)
            overall_quality_score = (
                (1 - quality_metrics['hallucination_rate']) * 0.3 +  # Lower hallucination is better
                quality_metrics['contextual_relevance'] * 0.3 +
                quality_metrics['answer_accuracy'] * 0.2 +
                quality_metrics['response_completeness'] * 0.2
            )
            
            # Human-like behavior score
            human_like_behavior_score = (
                behavior_metrics['persona_consistency'] * 0.4 +
                behavior_metrics['engagement_level'] * 0.3 +
                (1 if behavior_metrics['clarification_questions_asked'] > 0 else 0.5) * 0.3
            )
            
            # Multi-agent architecture score
            multi_agent_architecture_score = sum(agent_metrics.values()) / len(agent_metrics)
            
            # Memory utilization score
            memory_utilization_score = sum(memory_metrics.values()) / len(memory_metrics)
            
            return {
                'overall_quality_score': overall_quality_score,
                'human_like_behavior_score': human_like_behavior_score,
                'multi_agent_architecture_score': multi_agent_architecture_score,
                'memory_utilization_score': memory_utilization_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall scores: {e}")
            return {
                'overall_quality_score': 0.5,
                'human_like_behavior_score': 0.5,
                'multi_agent_architecture_score': 0.5,
                'memory_utilization_score': 0.5
            }
    
    # Helper methods for individual metric calculations
    async def _detect_hallucination_rate(self, response: str, response_metadata: Dict[str, Any]) -> float:
        """Detect hallucination rate in response"""
        try:
            # Simple heuristic-based detection
            unsupported_claims = ['definitely', 'certainly', 'always', 'never', 'proven', 'guaranteed']
            response_lower = response.lower()
            
            unsupported_count = sum(1 for claim in unsupported_claims if claim in response_lower)
            total_words = len(response.split())
            
            if total_words == 0:
                return 0.0
            
            # Normalize to 0-1 scale
            hallucination_rate = min(1.0, unsupported_count / (total_words / 100))
            return hallucination_rate
            
        except Exception as e:
            logger.error(f"Error detecting hallucination rate: {e}")
            return 0.5
    
    async def _calculate_contextual_relevance(self, query: str, response: str) -> float:
        """Calculate contextual relevance between query and response"""
        try:
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            if not query_words:
                return 0.0
            
            overlap = len(query_words.intersection(response_words))
            relevance = overlap / len(query_words)
            
            return min(1.0, relevance)
            
        except Exception as e:
            logger.error(f"Error calculating contextual relevance: {e}")
            return 0.5
    
    async def _calculate_response_completeness(self, query: str, response: str) -> float:
        """Calculate response completeness"""
        try:
            # Simple heuristic: longer responses for complex queries are more complete
            query_complexity = len(query.split())
            response_length = len(response.split())
            
            if query_complexity == 0:
                return 0.5
            
            # Normalize based on query complexity
            completeness = min(1.0, response_length / (query_complexity * 3))
            return completeness
            
        except Exception as e:
            logger.error(f"Error calculating response completeness: {e}")
            return 0.5
    
    async def _check_persona_consistency(self, response: str) -> float:
        """Check persona consistency in response"""
        try:
            # Check for persona indicators
            persona_indicators = ['sophia', 'i am', 'i have', 'i believe', 'i think', 'i would']
            response_lower = response.lower()
            
            persona_count = sum(1 for indicator in persona_indicators if indicator in response_lower)
            
            # Normalize to 0-1 scale
            consistency = min(1.0, persona_count / 3)
            return consistency
            
        except Exception as e:
            logger.error(f"Error checking persona consistency: {e}")
            return 0.5
    
    async def _calculate_engagement_level(self, response: str) -> float:
        """Calculate engagement level of response"""
        try:
            # Check for engaging elements
            engaging_elements = ['?', '!', 'interesting', 'fascinating', 'great question', 'let me explain']
            response_lower = response.lower()
            
            engagement_count = sum(1 for element in engaging_elements if element in response_lower)
            
            # Normalize to 0-1 scale
            engagement = min(1.0, engagement_count / 5)
            return engagement
            
        except Exception as e:
            logger.error(f"Error calculating engagement level: {e}")
            return 0.5
    
    async def _calculate_domain_specificity(self, query: str, response: str) -> float:
        """Calculate domain specificity (GenAI focus)"""
        try:
            genai_keywords = [
                'ai', 'artificial intelligence', 'machine learning', 'llm', 'gpt', 'claude',
                'generative', 'transformer', 'neural network', 'nlp', 'computer vision',
                'rag', 'vector database', 'embedding', 'fine-tuning', 'prompt engineering'
            ]
            
            query_lower = query.lower()
            response_lower = response.lower()
            
            query_genai_count = sum(1 for keyword in genai_keywords if keyword in query_lower)
            response_genai_count = sum(1 for keyword in genai_keywords if keyword in response_lower)
            
            # Calculate specificity
            if query_genai_count > 0:
                specificity = min(1.0, response_genai_count / query_genai_count)
            else:
                specificity = 0.5  # Neutral if no GenAI keywords in query
            
            return specificity
            
        except Exception as e:
            logger.error(f"Error calculating domain specificity: {e}")
            return 0.5
    
    async def _calculate_technical_accuracy(self, response: str, response_metadata: Dict[str, Any]) -> float:
        """Calculate technical accuracy"""
        try:
            # Use confidence score as proxy for technical accuracy
            confidence = response_metadata.get('confidence', 0.5)
            
            # Check for technical terms and citations
            technical_indicators = ['according to', 'research shows', 'studies indicate', 'source:', 'reference:']
            response_lower = response.lower()
            
            technical_count = sum(1 for indicator in technical_indicators if indicator in response_lower)
            technical_bonus = min(0.2, technical_count * 0.1)
            
            accuracy = min(1.0, confidence + technical_bonus)
            return accuracy
            
        except Exception as e:
            logger.error(f"Error calculating technical accuracy: {e}")
            return 0.5
    
    async def _calculate_practical_utility(self, response: str) -> float:
        """Calculate practical utility of response"""
        try:
            # Check for actionable elements
            actionable_indicators = [
                'how to', 'step by step', 'tutorial', 'guide', 'recommendation',
                'best practice', 'example', 'implementation', 'code', 'command'
            ]
            
            response_lower = response.lower()
            actionable_count = sum(1 for indicator in actionable_indicators if indicator in response_lower)
            
            # Normalize to 0-1 scale
            utility = min(1.0, actionable_count / 3)
            return utility
            
        except Exception as e:
            logger.error(f"Error calculating practical utility: {e}")
            return 0.5
    
    async def _calculate_response_continuity(self, response_metadata: Dict[str, Any]) -> float:
        """Calculate response continuity"""
        try:
            # Check if response builds on previous context
            memory_context = response_metadata.get('memory_context', {})
            conversation_history = memory_context.get('conversation_history', [])
            
            if len(conversation_history) > 0:
                continuity = 0.8  # High continuity if using conversation history
            else:
                continuity = 0.5  # Medium continuity if no history
            
            return continuity
            
        except Exception as e:
            logger.error(f"Error calculating response continuity: {e}")
            return 0.5
    
    def _create_default_metrics(self, query: str, response: str, session_id: str) -> EvaluationMetrics:
        """Create default metrics when calculation fails"""
        return EvaluationMetrics(
            query=query,
            response=response,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            hallucination_rate=0.5,
            contextual_relevance=0.5,
            answer_accuracy=0.5,
            response_completeness=0.5,
            retrieval_agent_score=0.5,
            webscraper_agent_score=0.5,
            reasoning_agent_score=0.5,
            memory_agent_score=0.5,
            evaluation_agent_score=0.5,
            persona_consistency=0.5,
            engagement_level=0.5,
            clarification_questions_asked=0,
            off_topic_redirects=0,
            short_term_memory_used=False,
            long_term_memory_used=False,
            user_preferences_utilized=False,
            conversation_context_used=False,
            domain_specificity=0.5,
            technical_accuracy=0.5,
            practical_utility=0.5,
            response_continuity=0.5,
            overall_quality_score=0.5,
            human_like_behavior_score=0.5,
            multi_agent_architecture_score=0.5,
            memory_utilization_score=0.5,
            response_time_seconds=0.0,
            token_usage=0,
            cost_estimate=0.0,
            evaluation_method='default'
        )


class CSVReporter:
    """Handles CSV reporting of evaluation metrics"""
    
    def __init__(self, metrics_collector: EvaluationMetricsCollector):
        self.metrics_collector = metrics_collector
    
    def export_comprehensive_report(self, filename: str = None) -> str:
        """Export comprehensive evaluation report to CSV"""
        try:
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"genai_chatbot_evaluation_report_{timestamp}.csv"
            
            # Convert metrics to DataFrame
            data = []
            for metrics in self.metrics_collector.metrics_history:
                data.append(asdict(metrics))
            
            if not data:
                logger.warning("No metrics data to export")
                return ""
            
            df = pd.DataFrame(data)
            
            # Export to CSV
            df.to_csv(filename, index=False)
            
            logger.info(f"Exported {len(data)} evaluation records to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting comprehensive report: {e}")
            return ""
    
    def export_summary_report(self, filename: str = None) -> str:
        """Export summary report with aggregated metrics"""
        try:
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"genai_chatbot_summary_report_{timestamp}.csv"
            
            if not self.metrics_collector.metrics_history:
                logger.warning("No metrics data for summary report")
                return ""
            
            # Calculate summary statistics
            summary_data = self._calculate_summary_statistics()
            
            # Create summary DataFrame
            df = pd.DataFrame([summary_data])
            
            # Export to CSV
            df.to_csv(filename, index=False)
            
            logger.info(f"Exported summary report to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting summary report: {e}")
            return ""
    
    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics from metrics history"""
        try:
            metrics = self.metrics_collector.metrics_history
            
            if not metrics:
                return {}
            
            # Calculate averages for all numeric metrics
            summary = {
                'total_evaluations': len(metrics),
                'evaluation_period_start': min(m.timestamp for m in metrics),
                'evaluation_period_end': max(m.timestamp for m in metrics),
                
                # Quality metrics averages
                'avg_hallucination_rate': sum(m.hallucination_rate for m in metrics) / len(metrics),
                'avg_contextual_relevance': sum(m.contextual_relevance for m in metrics) / len(metrics),
                'avg_answer_accuracy': sum(m.answer_accuracy for m in metrics) / len(metrics),
                'avg_response_completeness': sum(m.response_completeness for m in metrics) / len(metrics),
                
                # Agent performance averages
                'avg_retrieval_agent_score': sum(m.retrieval_agent_score for m in metrics) / len(metrics),
                'avg_webscraper_agent_score': sum(m.webscraper_agent_score for m in metrics) / len(metrics),
                'avg_reasoning_agent_score': sum(m.reasoning_agent_score for m in metrics) / len(metrics),
                'avg_memory_agent_score': sum(m.memory_agent_score for m in metrics) / len(metrics),
                'avg_evaluation_agent_score': sum(m.evaluation_agent_score for m in metrics) / len(metrics),
                
                # Human-like behavior averages
                'avg_persona_consistency': sum(m.persona_consistency for m in metrics) / len(metrics),
                'avg_engagement_level': sum(m.engagement_level for m in metrics) / len(metrics),
                'total_clarification_questions': sum(m.clarification_questions_asked for m in metrics),
                'total_off_topic_redirects': sum(m.off_topic_redirects for m in metrics),
                
                # Memory utilization percentages
                'short_term_memory_usage_rate': sum(1 for m in metrics if m.short_term_memory_used) / len(metrics),
                'long_term_memory_usage_rate': sum(1 for m in metrics if m.long_term_memory_used) / len(metrics),
                'user_preferences_usage_rate': sum(1 for m in metrics if m.user_preferences_utilized) / len(metrics),
                'conversation_context_usage_rate': sum(1 for m in metrics if m.conversation_context_used) / len(metrics),
                
                # Custom evaluation matrix averages
                'avg_domain_specificity': sum(m.domain_specificity for m in metrics) / len(metrics),
                'avg_technical_accuracy': sum(m.technical_accuracy for m in metrics) / len(metrics),
                'avg_practical_utility': sum(m.practical_utility for m in metrics) / len(metrics),
                'avg_response_continuity': sum(m.response_continuity for m in metrics) / len(metrics),
                
                # Overall scores averages
                'avg_overall_quality_score': sum(m.overall_quality_score for m in metrics) / len(metrics),
                'avg_human_like_behavior_score': sum(m.human_like_behavior_score for m in metrics) / len(metrics),
                'avg_multi_agent_architecture_score': sum(m.multi_agent_architecture_score for m in metrics) / len(metrics),
                'avg_memory_utilization_score': sum(m.memory_utilization_score for m in metrics) / len(metrics),
                
                # Performance metrics
                'avg_response_time': sum(m.response_time_seconds for m in metrics) / len(metrics),
                'total_token_usage': sum(m.token_usage for m in metrics),
                'total_cost_estimate': sum(m.cost_estimate for m in metrics),
                'avg_cost_per_query': sum(m.cost_estimate for m in metrics) / len(metrics)
            }
            
            # Calculate pass rates based on thresholds
            thresholds = self.metrics_collector.evaluation_thresholds
            summary.update({
                'hallucination_rate_pass_rate': sum(1 for m in metrics if m.hallucination_rate <= thresholds['hallucination_rate']) / len(metrics),
                'contextual_relevance_pass_rate': sum(1 for m in metrics if m.contextual_relevance >= thresholds['contextual_relevance']) / len(metrics),
                'overall_quality_pass_rate': sum(1 for m in metrics if m.overall_quality_score >= thresholds['overall_quality_score']) / len(metrics),
                'human_like_behavior_pass_rate': sum(1 for m in metrics if m.human_like_behavior_score >= thresholds['human_like_behavior_score']) / len(metrics),
                'multi_agent_architecture_pass_rate': sum(1 for m in metrics if m.multi_agent_architecture_score >= thresholds['multi_agent_architecture_score']) / len(metrics),
                'memory_utilization_pass_rate': sum(1 for m in metrics if m.memory_utilization_score >= thresholds['memory_utilization_score']) / len(metrics)
            })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {e}")
            return {}
    
    def generate_evaluation_matrix_report(self, filename: str = None) -> str:
        """Generate evaluation matrix report as specified in requirements"""
        try:
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"evaluation_matrix_report_{timestamp}.csv"
            
            if not self.metrics_collector.metrics_history:
                logger.warning("No metrics data for evaluation matrix report")
                return ""
            
            # Create evaluation matrix data
            matrix_data = []
            for i, metrics in enumerate(self.metrics_collector.metrics_history):
                matrix_data.append({
                    'query_id': i + 1,
                    'query': metrics.query,
                    'hallucination_rate': metrics.hallucination_rate,
                    'contextual_relevance': metrics.contextual_relevance,
                    'use_of_memory': metrics.memory_utilization_score,
                    'human_like_behavior': metrics.human_like_behavior_score,
                    'domain_specificity': metrics.domain_specificity,
                    'accuracy': metrics.technical_accuracy,
                    'engagement': metrics.engagement_level,
                    'continuity': metrics.response_continuity,
                    'overall_score': metrics.overall_quality_score,
                    'timestamp': metrics.timestamp
                })
            
            # Create DataFrame and export
            df = pd.DataFrame(matrix_data)
            df.to_csv(filename, index=False)
            
            logger.info(f"Exported evaluation matrix report to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error generating evaluation matrix report: {e}")
            return ""


# Global instances
metrics_collector = EvaluationMetricsCollector()
csv_reporter = CSVReporter(metrics_collector)
