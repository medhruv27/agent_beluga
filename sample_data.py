"""
Sample data and test queries for the GenAI chatbot
"""
from typing import List, Dict, Any
from datetime import datetime


# Sample test queries for evaluation
SAMPLE_QUERIES = [
    # Basic GenAI concepts
    "What is a large language model?",
    "Explain the difference between GPT and BERT",
    "How do transformers work in NLP?",
    "What is the attention mechanism?",
    
    # Framework and tools
    "Compare LangChain vs LangGraph for building AI agents",
    "How do I implement RAG with vector databases?",
    "What are the best practices for prompt engineering?",
    "How to fine-tune a language model?",
    
    # Technical implementations
    "Show me how to build a chatbot with memory",
    "How do I implement semantic search?",
    "What's the best way to handle context windows?",
    "How to optimize LLM inference speed?",
    
    # Research and trends
    "What are the latest trends in multimodal AI?",
    "How is AI being used in code generation?",
    "What are the current limitations of LLMs?",
    "What's new in AI safety research?",
    
    # Comparisons and analysis
    "Compare OpenAI GPT-4 vs Google Gemini Pro",
    "What are the pros and cons of open-source LLMs?",
    "How do different embedding models compare?",
    "Which vector database is best for RAG?",
    
    # Practical applications
    "How to build a document Q&A system?",
    "What's the best approach for AI agent orchestration?",
    "How to implement few-shot learning?",
    "What are the best practices for AI evaluation?",
    
    # Advanced topics
    "Explain the architecture of modern LLMs",
    "How does reinforcement learning from human feedback work?",
    "What is chain-of-thought prompting?",
    "How to implement retrieval-augmented generation?",
    
    # Industry and market
    "What AI startups are worth watching?",
    "How is the AI job market evolving?",
    "What are the latest AI funding trends?",
    "Which companies are leading in AI research?",
    
    # Ethics and safety
    "What are the ethical concerns with LLMs?",
    "How to prevent AI hallucination?",
    "What is AI alignment and why is it important?",
    "How to ensure AI safety in production?",
    
    # Future and speculation
    "What will AI look like in 5 years?",
    "How will AI change software development?",
    "What are the next breakthroughs in AI?",
    "How will AI impact education?",
    
    # Troubleshooting and optimization
    "My LLM responses are too generic, how to improve?",
    "How to reduce token costs in production?",
    "What causes AI model drift?",
    "How to handle rate limits with AI APIs?",
    
    # Integration and deployment
    "How to deploy LLMs in production?",
    "What's the best way to monitor AI systems?",
    "How to handle AI model versioning?",
    "What are the security considerations for AI apps?",
    
    # Off-topic queries (for testing redirects)
    "What's the weather like today?",
    "How do I cook pasta?",
    "What's the capital of France?",
    "Tell me a joke"
]


# Sample documents for vector database
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "Introduction to Large Language Models",
        "content": "Large Language Models (LLMs) are neural networks trained on vast amounts of text data to understand and generate human-like text. They use transformer architecture with attention mechanisms to process sequential data efficiently. Popular examples include GPT, BERT, and T5 models.",
        "source": "AI Research Paper",
        "url": "https://example.com/llm-intro",
        "metadata": {
            "category": "fundamentals",
            "difficulty": "beginner",
            "tags": ["llm", "transformer", "nlp"]
        }
    },
    {
        "id": "doc_2", 
        "title": "RAG Implementation Guide",
        "content": "Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval. It involves creating embeddings of documents, storing them in a vector database, and retrieving relevant context during generation. This approach reduces hallucination and provides more accurate, up-to-date information.",
        "source": "Technical Blog",
        "url": "https://example.com/rag-guide",
        "metadata": {
            "category": "implementation",
            "difficulty": "intermediate",
            "tags": ["rag", "vector-database", "embeddings"]
        }
    },
    {
        "id": "doc_3",
        "title": "LangChain vs LangGraph Comparison",
        "content": "LangChain is a framework for building applications with LLMs, focusing on chaining components together. LangGraph extends LangChain with graph-based workflows, enabling more complex agent behaviors with cycles, conditional logic, and state management. Choose LangChain for simple chains, LangGraph for complex agent workflows.",
        "source": "Framework Documentation",
        "url": "https://example.com/langchain-vs-langgraph",
        "metadata": {
            "category": "frameworks",
            "difficulty": "intermediate",
            "tags": ["langchain", "langgraph", "agents", "workflows"]
        }
    },
    {
        "id": "doc_4",
        "title": "Prompt Engineering Best Practices",
        "content": "Effective prompt engineering involves crafting clear, specific instructions that guide the LLM to produce desired outputs. Key techniques include few-shot learning, chain-of-thought prompting, role-based prompting, and iterative refinement. Always test prompts with various inputs and adjust based on results.",
        "source": "AI Best Practices Guide",
        "url": "https://example.com/prompt-engineering",
        "metadata": {
            "category": "techniques",
            "difficulty": "beginner",
            "tags": ["prompt-engineering", "best-practices", "optimization"]
        }
    },
    {
        "id": "doc_5",
        "title": "Multimodal AI Applications",
        "content": "Multimodal AI systems can process and generate content across different modalities like text, images, audio, and video. Applications include image captioning, visual question answering, video understanding, and cross-modal retrieval. Recent advances in models like GPT-4V and CLIP have significantly improved multimodal capabilities.",
        "source": "Research Review",
        "url": "https://example.com/multimodal-ai",
        "metadata": {
            "category": "applications",
            "difficulty": "advanced",
            "tags": ["multimodal", "computer-vision", "applications"]
        }
    }
]


# Sample user preferences for testing memory
SAMPLE_USER_PREFERENCES = {
    "beginner_user": {
        "interests": ["basics", "tutorials", "simple-explanations"],
        "expertise_level": "beginner",
        "preferred_topics": ["introduction", "fundamentals", "getting-started"],
        "interaction_style": "detailed",
        "last_updated": datetime.now().isoformat()
    },
    "intermediate_user": {
        "interests": ["implementation", "frameworks", "best-practices"],
        "expertise_level": "intermediate", 
        "preferred_topics": ["rag", "agents", "optimization"],
        "interaction_style": "balanced",
        "last_updated": datetime.now().isoformat()
    },
    "advanced_user": {
        "interests": ["research", "architecture", "advanced-techniques"],
        "expertise_level": "advanced",
        "preferred_topics": ["multimodal", "fine-tuning", "custom-models"],
        "interaction_style": "concise",
        "last_updated": datetime.now().isoformat()
    }
}


# Sample conversation history for testing memory
SAMPLE_CONVERSATION_HISTORY = [
    {
        "timestamp": "2024-01-15T10:00:00",
        "user_query": "What is a transformer model?",
        "bot_response": "A transformer model is a neural network architecture that uses attention mechanisms to process sequential data. It was introduced in the paper 'Attention Is All You Need' and forms the foundation of modern LLMs like GPT and BERT.",
        "context": {
            "sources": [{"title": "Transformer Paper", "url": "https://example.com/transformer"}],
            "confidence": 0.9
        }
    },
    {
        "timestamp": "2024-01-15T10:05:00",
        "user_query": "How does attention work in transformers?",
        "bot_response": "Attention in transformers allows the model to focus on different parts of the input sequence when processing each position. It computes relationships between all positions simultaneously, enabling parallel processing and capturing long-range dependencies effectively.",
        "context": {
            "sources": [{"title": "Attention Mechanism Guide", "url": "https://example.com/attention"}],
            "confidence": 0.85
        }
    }
]


# Sample evaluation results for testing
SAMPLE_EVALUATION_RESULTS = [
    {
        "query": "What is machine learning?",
        "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
        "hallucination_rate": 0.1,
        "contextual_relevance": 0.9,
        "answer_accuracy": 0.8,
        "response_completeness": 0.7,
        "overall_quality_score": 0.8,
        "human_like_behavior_score": 0.7,
        "multi_agent_architecture_score": 0.8,
        "memory_utilization_score": 0.6,
        "timestamp": datetime.now().isoformat()
    },
    {
        "query": "How do I implement RAG?",
        "response": "To implement RAG, you'll need to: 1) Create embeddings of your documents, 2) Store them in a vector database, 3) Retrieve relevant documents for each query, 4) Pass the context to your LLM for generation.",
        "hallucination_rate": 0.05,
        "contextual_relevance": 0.95,
        "answer_accuracy": 0.9,
        "response_completeness": 0.8,
        "overall_quality_score": 0.9,
        "human_like_behavior_score": 0.8,
        "multi_agent_architecture_score": 0.85,
        "memory_utilization_score": 0.7,
        "timestamp": datetime.now().isoformat()
    }
]


def get_sample_queries_by_category() -> Dict[str, List[str]]:
    """Get sample queries organized by category"""
    return {
        "basic_concepts": SAMPLE_QUERIES[:4],
        "frameworks_tools": SAMPLE_QUERIES[4:8],
        "technical_implementation": SAMPLE_QUERIES[8:12],
        "research_trends": SAMPLE_QUERIES[12:16],
        "comparisons_analysis": SAMPLE_QUERIES[16:20],
        "practical_applications": SAMPLE_QUERIES[20:24],
        "advanced_topics": SAMPLE_QUERIES[24:28],
        "industry_market": SAMPLE_QUERIES[28:32],
        "ethics_safety": SAMPLE_QUERIES[32:36],
        "future_speculation": SAMPLE_QUERIES[36:40],
        "troubleshooting": SAMPLE_QUERIES[40:44],
        "integration_deployment": SAMPLE_QUERIES[44:48],
        "off_topic": SAMPLE_QUERIES[48:52]
    }


def get_evaluation_test_suite() -> List[Dict[str, Any]]:
    """Get comprehensive test suite for evaluation"""
    test_suite = []
    
    for i, query in enumerate(SAMPLE_QUERIES[:50]):  # Use first 50 queries
        test_suite.append({
            "query_id": i + 1,
            "query": query,
            "expected_category": "genai" if i < 48 else "off_topic",
            "expected_clarification": i in [1, 5, 9, 13, 17],  # Some queries should trigger clarification
            "expected_sources": i < 48,  # GenAI queries should have sources
            "difficulty": "beginner" if i < 12 else "intermediate" if i < 36 else "advanced"
        })
    
    return test_suite


if __name__ == "__main__":
    # Print sample data for verification
    print("Sample Queries:", len(SAMPLE_QUERIES))
    print("Sample Documents:", len(SAMPLE_DOCUMENTS))
    print("Sample User Preferences:", len(SAMPLE_USER_PREFERENCES))
    print("Sample Conversation History:", len(SAMPLE_CONVERSATION_HISTORY))
    print("Sample Evaluation Results:", len(SAMPLE_EVALUATION_RESULTS))
    
    # Print categorized queries
    categories = get_sample_queries_by_category()
    for category, queries in categories.items():
        print(f"{category}: {len(queries)} queries")
    
    # Print test suite info
    test_suite = get_evaluation_test_suite()
    print(f"Evaluation Test Suite: {len(test_suite)} test cases")
