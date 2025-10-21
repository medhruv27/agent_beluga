"""
Configuration settings for the GenAI Chatbot
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Qdrant Configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    
    # LangSmith Configuration
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "genai-chatbot"
    
    # Web Scraping
    serpapi_key: Optional[str] = None
    
    # Application Settings
    app_name: str = "GenAI Research Assistant"
    debug: bool = True
    log_level: str = "INFO"
    
    # Model Settings
    primary_llm: str = "gemini-2.0-flash"  # For complex queries
    secondary_llm: str = "gemini-2.0-flash"  # For simple tasks
    embedding_model: str = "models/gemini-embedding-001"  # Gemini embedding model
    
    # Memory Settings
    max_conversation_history: int = 10
    memory_retention_days: int = 30
    
    # Evaluation Settings
    evaluation_threshold: float = 0.8
    max_evaluation_queries: int = 50
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
