"""
Test suite for evaluation metrics and CSV reporting
"""
import pytest
import asyncio
from datetime import datetime
from evaluation_metrics import EvaluationMetricsCollector, CSVReporter, EvaluationMetrics


class TestEvaluationMetrics:
    """Test evaluation metrics collection"""
    
    @pytest.fixture
    def metrics_collector(self):
        return EvaluationMetricsCollector()
    
    @pytest.fixture
    def sample_query_response(self):
        return {
            "query": "What are the latest trends in large language models?",
            "response": "Large language models are evolving rapidly with several key trends: 1) Multimodal capabilities combining text, images, and audio, 2) Smaller, more efficient models like LLaMA and Mistral, 3) Open-source alternatives gaining traction, 4) Improved reasoning and code generation capabilities.",
            "session_id": "test-session-123",
            "agent_performance": {
                "retrieval_score": 0.8,
                "webscraper_score": 0.7,
                "reasoning_score": 0.9,
                "memory_score": 0.6,
                "evaluation_score": 0.8
            },
            "response_metadata": {
                "confidence": 0.85,
                "sources": [
                    {"title": "LLM Trends 2024", "source": "Research Paper", "url": "https://example.com"}
                ],
                "memory_context": {
                    "conversation_history": [{"type": "user", "content": "Previous question"}],
                    "relevant_memories": [{"content": "Previous relevant info"}],
                    "user_preferences": {"expertise_level": "intermediate"}
                },
                "response_time": 2.5,
                "token_usage": 150,
                "cost_estimate": 0.01
            }
        }
    
    @pytest.mark.asyncio
    async def test_collect_comprehensive_metrics(self, metrics_collector, sample_query_response):
        """Test comprehensive metrics collection"""
        metrics = await metrics_collector.collect_comprehensive_metrics(
            query=sample_query_response["query"],
            response=sample_query_response["response"],
            session_id=sample_query_response["session_id"],
            agent_performance=sample_query_response["agent_performance"],
            response_metadata=sample_query_response["response_metadata"]
        )
        
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.query == sample_query_response["query"]
        assert metrics.response == sample_query_response["response"]
        assert metrics.session_id == sample_query_response["session_id"]
        assert 0 <= metrics.hallucination_rate <= 1
        assert 0 <= metrics.contextual_relevance <= 1
        assert 0 <= metrics.overall_quality_score <= 1
    
    @pytest.mark.asyncio
    async def test_quality_metrics_calculation(self, metrics_collector):
        """Test quality metrics calculation"""
        query = "What is machine learning?"
        response = "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
        metadata = {"confidence": 0.9}
        
        quality_metrics = await metrics_collector._calculate_quality_metrics(query, response, metadata)
        
        assert "hallucination_rate" in quality_metrics
        assert "contextual_relevance" in quality_metrics
        assert "answer_accuracy" in quality_metrics
        assert "response_completeness" in quality_metrics
        
        for metric in quality_metrics.values():
            assert 0 <= metric <= 1
    
    @pytest.mark.asyncio
    async def test_behavior_metrics_calculation(self, metrics_collector):
        """Test human-like behavior metrics calculation"""
        query = "How do I implement RAG?"
        response = "I'd be happy to help you implement RAG! Let me ask a few questions to better understand your specific needs..."
        metadata = {"requires_clarification": True, "type": "clarification"}
        
        behavior_metrics = await metrics_collector._calculate_behavior_metrics(query, response, metadata)
        
        assert "persona_consistency" in behavior_metrics
        assert "engagement_level" in behavior_metrics
        assert "clarification_questions_asked" in behavior_metrics
        assert "off_topic_redirects" in behavior_metrics
        
        assert behavior_metrics["clarification_questions_asked"] == 1


class TestCSVReporter:
    """Test CSV reporting functionality"""
    
    @pytest.fixture
    def metrics_collector(self):
        return EvaluationMetricsCollector()
    
    @pytest.fixture
    def csv_reporter(self, metrics_collector):
        return CSVReporter(metrics_collector)
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing"""
        return EvaluationMetrics(
            query="Test query",
            response="Test response",
            timestamp=datetime.now().isoformat(),
            session_id="test-session",
            hallucination_rate=0.1,
            contextual_relevance=0.9,
            answer_accuracy=0.8,
            response_completeness=0.7,
            retrieval_agent_score=0.8,
            webscraper_agent_score=0.7,
            reasoning_agent_score=0.9,
            memory_agent_score=0.6,
            evaluation_agent_score=0.8,
            persona_consistency=0.8,
            engagement_level=0.7,
            clarification_questions_asked=1,
            off_topic_redirects=0,
            short_term_memory_used=True,
            long_term_memory_used=False,
            user_preferences_utilized=True,
            conversation_context_used=True,
            domain_specificity=0.9,
            technical_accuracy=0.8,
            practical_utility=0.7,
            response_continuity=0.8,
            overall_quality_score=0.8,
            human_like_behavior_score=0.7,
            multi_agent_architecture_score=0.8,
            memory_utilization_score=0.75,
            response_time_seconds=2.5,
            token_usage=150,
            cost_estimate=0.01,
            evaluation_method="test"
        )
    
    def test_export_comprehensive_report(self, csv_reporter, sample_metrics, tmp_path):
        """Test comprehensive report export"""
        # Add sample metrics to collector
        csv_reporter.metrics_collector.metrics_history.append(sample_metrics)
        
        # Export report
        filename = tmp_path / "test_report.csv"
        result = csv_reporter.export_comprehensive_report(str(filename))
        
        assert result == str(filename)
        assert filename.exists()
        
        # Verify CSV content
        import pandas as pd
        df = pd.read_csv(filename)
        assert len(df) == 1
        assert df.iloc[0]['query'] == "Test query"
        assert df.iloc[0]['response'] == "Test response"
    
    def test_export_summary_report(self, csv_reporter, sample_metrics, tmp_path):
        """Test summary report export"""
        # Add sample metrics to collector
        csv_reporter.metrics_collector.metrics_history.append(sample_metrics)
        
        # Export summary report
        filename = tmp_path / "test_summary.csv"
        result = csv_reporter.export_summary_report(str(filename))
        
        assert result == str(filename)
        assert filename.exists()
        
        # Verify CSV content
        import pandas as pd
        df = pd.read_csv(filename)
        assert len(df) == 1
        assert 'total_evaluations' in df.columns
        assert df.iloc[0]['total_evaluations'] == 1
    
    def test_evaluation_matrix_report(self, csv_reporter, sample_metrics, tmp_path):
        """Test evaluation matrix report export"""
        # Add sample metrics to collector
        csv_reporter.metrics_collector.metrics_history.append(sample_metrics)
        
        # Export evaluation matrix report
        filename = tmp_path / "test_matrix.csv"
        result = csv_reporter.generate_evaluation_matrix_report(str(filename))
        
        assert result == str(filename)
        assert filename.exists()
        
        # Verify CSV content
        import pandas as pd
        df = pd.read_csv(filename)
        assert len(df) == 1
        assert 'query_id' in df.columns
        assert 'hallucination_rate' in df.columns
        assert 'contextual_relevance' in df.columns
        assert 'overall_score' in df.columns


class TestEvaluationThresholds:
    """Test evaluation threshold functionality"""
    
    def test_evaluation_thresholds(self):
        """Test that evaluation thresholds are properly defined"""
        collector = EvaluationMetricsCollector()
        
        required_thresholds = [
            'hallucination_rate',
            'contextual_relevance',
            'overall_quality_score',
            'human_like_behavior_score',
            'multi_agent_architecture_score',
            'memory_utilization_score'
        ]
        
        for threshold in required_thresholds:
            assert threshold in collector.evaluation_thresholds
            assert 0 <= collector.evaluation_thresholds[threshold] <= 1


if __name__ == "__main__":
    pytest.main([__file__])
