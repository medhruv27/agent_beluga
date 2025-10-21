"""
FastAPI backend for the GenAI Chatbot with multi-agent orchestration
"""
import asyncio
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from datetime import datetime
from config import settings
print(settings.google_api_key)  # Should print your Gemini API key if loaded correctly

# Import agents
from agents import (
    RetrievalAgent, WebScraperAgent, ReasoningAgent, 
    MemoryAgent, EvaluationAgent
)
from persona import GenAIResearcherPersona
from config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="A sophisticated human-like chatbot specializing in Generative AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instances
retrieval_agent = None
webscraper_agent = None
reasoning_agent = None
memory_agent = None
evaluation_agent = None
persona = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: list
    confidence: float
    requires_clarification: bool
    timestamp: str
    evaluation_score: Optional[float] = None


class AgentStatus(BaseModel):
    agent: str
    status: str
    last_updated: str


@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup"""
    global retrieval_agent, webscraper_agent, reasoning_agent, memory_agent, evaluation_agent, persona
    
    try:
        logger.info("Initializing GenAI Chatbot agents...")
        
        # Initialize persona
        persona = GenAIResearcherPersona()
        
        # Initialize agents
        retrieval_agent = RetrievalAgent()
        webscraper_agent = WebScraperAgent()
        reasoning_agent = ReasoningAgent()
        memory_agent = MemoryAgent()
        evaluation_agent = EvaluationAgent()
        
        logger.info("All agents initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing agents: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "persona": persona.persona.name if persona else "Not initialized",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check agent status
        agent_statuses = []
        
        if retrieval_agent:
            agent_statuses.append(AgentStatus(
                agent="retrieval",
                status="active",
                last_updated=datetime.now().isoformat()
            ))
        
        if webscraper_agent:
            agent_statuses.append(AgentStatus(
                agent="webscraper",
                status="active", 
                last_updated=datetime.now().isoformat()
            ))
        
        if reasoning_agent:
            agent_statuses.append(AgentStatus(
                agent="reasoning",
                status="active",
                last_updated=datetime.now().isoformat()
            ))
        
        if memory_agent:
            agent_statuses.append(AgentStatus(
                agent="memory",
                status="active",
                last_updated=datetime.now().isoformat()
            ))
        
        if evaluation_agent:
            agent_statuses.append(AgentStatus(
                agent="evaluation",
                status="active",
                last_updated=datetime.now().isoformat()
            ))
        
        return {
            "status": "healthy",
            "agents": agent_statuses,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with multi-agent orchestration"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Processing chat request for session {session_id}")
        
        # Orchestrate multi-agent workflow
        response_data = await orchestrate_agents(
            message=request.message,
            session_id=session_id,
            user_context=request.user_context or {}
        )
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Error processing chat request")


async def orchestrate_agents(
    message: str,
    session_id: str,
    user_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Orchestrate the multi-agent workflow"""
    try:
        # 1. Memory Agent: Get context and user preferences
        memory_context = await memory_agent.get_memory_context(session_id, message)
        
        # 2. Check if clarification is needed
        requires_clarification = persona.should_ask_clarifying_questions(message)
        
        if requires_clarification:
            # Generate clarifying questions
            response = await reasoning_agent.generate_response(
                user_query=message,
                requires_clarification=True
            )
            
            return {
                'response': response['response'],
                'session_id': session_id,
                'sources': [],
                'confidence': response['confidence'],
                'requires_clarification': True,
                'timestamp': response['timestamp'],
                'evaluation_score': None
            }
        
        # 3. Retrieval Agent: Search for relevant information
        retrieved_info = await retrieval_agent.hybrid_search(
            query=message,
            limit=5
        )
        
        # 4. Web Scraper Agent: Get latest updates (if needed)
        web_info = []
        if await should_scrape_web(message):
            async with webscraper_agent as scraper:
                web_info = await scraper.scrape_latest_updates(max_articles=3)
        
        # 5. Reasoning Agent: Generate response
        reasoning_response = await reasoning_agent.generate_response(
            user_query=message,
            retrieved_info=retrieved_info,
            web_info=web_info,
            memory_context=memory_context
        )
        
        # 6. Evaluation Agent: Evaluate response quality
        evaluation_result = await evaluation_agent.evaluate_response(
            query=message,
            response=reasoning_response['response'],
            sources=reasoning_response['sources'],
            context=[doc.get('content', '') for doc in retrieved_info]
        )
        
        # 7. Memory Agent: Store conversation
        await memory_agent.store_conversation_turn(
            session_id=session_id,
            user_query=message,
            bot_response=reasoning_response['response'],
            context={
                'sources': reasoning_response['sources'],
                'confidence': reasoning_response['confidence'],
                'evaluation_score': evaluation_result.overall_score
            }
        )
        
        return {
            'response': reasoning_response['response'],
            'session_id': session_id,
            'sources': reasoning_response['sources'],
            'confidence': reasoning_response['confidence'],
            'requires_clarification': reasoning_response.get('requires_follow_up', False),
            'timestamp': reasoning_response['timestamp'],
            'evaluation_score': evaluation_result.overall_score
        }
        
    except Exception as e:
        logger.error(f"Error in agent orchestration: {e}")
        # Return fallback response
        return {
            'response': f"I apologize, but I'm having trouble processing your request right now. As {persona.persona.name}, I'd be happy to help you with Generative AI questions once I'm back to full capacity!",
            'session_id': session_id,
            'sources': [],
            'confidence': 0.0,
            'requires_clarification': False,
            'timestamp': datetime.now().isoformat(),
            'evaluation_score': None
        }


async def should_scrape_web(message: str) -> bool:
    """Determine if web scraping is needed based on the message"""
    web_scraping_indicators = [
        'latest', 'recent', 'new', 'current', 'today', 'this week',
        'trending', 'news', 'updates', 'announcement'
    ]
    
    message_lower = message.lower()
    return any(indicator in message_lower for indicator in web_scraping_indicators)


@app.get("/persona")
async def get_persona():
    """Get persona information"""
    if not persona:
        raise HTTPException(status_code=500, detail="Persona not initialized")
    
    return {
        'name': persona.persona.name,
        'title': persona.persona.title,
        'background': persona.persona.background,
        'interests': persona.persona.interests,
        'expertise_areas': persona.persona.expertise_areas,
        'introduction': persona.get_introduction()
    }


@app.get("/memory/{session_id}")
async def get_memory(session_id: str):
    """Get memory context for a session"""
    try:
        memory_context = await memory_agent.get_memory_context(session_id, "")
        return memory_context
    except Exception as e:
        logger.error(f"Error getting memory: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving memory")


@app.get("/evaluation/summary")
async def get_evaluation_summary():
    """Get evaluation summary"""
    try:
        summary = evaluation_agent.get_evaluation_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting evaluation summary: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving evaluation summary")


@app.post("/evaluation/export")
async def export_evaluation_results():
    """Export evaluation results to CSV"""
    try:
        filename = evaluation_agent.export_evaluation_results()
        return {"filename": filename, "message": "Evaluation results exported successfully"}
    except Exception as e:
        logger.error(f"Error exporting evaluation results: {e}")
        raise HTTPException(status_code=500, detail="Error exporting evaluation results")


@app.get("/retrieval/stats")
async def get_retrieval_stats():
    """Get retrieval agent statistics"""
    try:
        stats = await retrieval_agent.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting retrieval stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving stats")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
