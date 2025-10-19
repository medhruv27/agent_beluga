"""
Reasoning Agent for synthesizing information and generating responses
"""
import asyncio
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import logging
from datetime import datetime
import json
from config import settings
from persona import GenAIResearcherPersona

logger = logging.getLogger(__name__)


class ReasoningAgent:
    """Handles reasoning and response generation using LLMs"""
    
    def __init__(self):
        self.persona = GenAIResearcherPersona()
        
        # Initialize LLMs
        self.primary_llm = ChatGoogleGenerativeAI(
            model=settings.primary_llm,
            google_api_key=settings.google_api_key,
            temperature=0.7,
            max_output_tokens=1000
        )
        
        self.secondary_llm = ChatGoogleGenerativeAI(
            model=settings.secondary_llm,
            google_api_key=settings.google_api_key,
            temperature=0.3,
            max_output_tokens=500
        )
        
        # Define prompt templates
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompt templates for different tasks"""
        
        # Main response generation prompt
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are {persona_name}, a {persona_title}. 

Background: {persona_background}
Expertise: {persona_expertise}
Personality: {persona_personality}
Interaction Style: {persona_interaction_style}

You specialize in Generative AI and must:
1. Always maintain your human persona
2. Ask clarifying questions when needed
3. Provide accurate, helpful information about GenAI
4. Redirect off-topic conversations back to GenAI
5. Use the retrieved information to inform your responses
6. Be conversational and engaging

Retrieved Information:
{retrieved_info}

Web Scraped Information:
{web_info}

Memory Context:
{memory_context}

User Query: {user_query}

Provide a helpful, accurate response that maintains your persona while addressing the user's question."""),
            ("human", "{user_query}")
        ])
        
        # Clarifying questions prompt
        self.clarification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are {persona_name}, a {persona_title}. 

You need to ask clarifying questions to better understand the user's intent before providing a detailed answer.

User Query: {user_query}

Generate 2-3 specific, helpful clarifying questions that will help you provide a more targeted and useful response. Be conversational and show genuine interest in understanding their needs.

Format your response as a list of questions, each on a new line."""),
            ("human", "{user_query}")
        ])
        
        # Information synthesis prompt
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in information synthesis. Your task is to combine and analyze information from multiple sources to create a coherent, accurate response.

Retrieved Documents:
{retrieved_docs}

Web Scraped Information:
{web_info}

User Query: {user_query}

Synthesize this information to provide:
1. Key insights and facts
2. Relevant examples and use cases
3. Potential limitations or considerations
4. Actionable recommendations

Be objective, accurate, and cite sources when possible."""),
            ("human", "{user_query}")
        ])
        
        # Off-topic redirect prompt
        self.redirect_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are {persona_name}, a {persona_title}. 

The user has asked about something outside your expertise in Generative AI. You need to politely redirect them back to GenAI topics while maintaining your friendly, helpful persona.

Off-topic Query: {user_query}

Generate a polite redirect message that:
1. Acknowledges their question
2. Explains your specialization in GenAI
3. Suggests related GenAI topics they might find interesting
4. Maintains a helpful, engaging tone"""),
            ("human", "{user_query}")
        ])
    
    async def generate_response(
        self,
        user_query: str,
        retrieved_info: List[Dict[str, Any]] = None,
        web_info: List[Dict[str, Any]] = None,
        memory_context: Dict[str, Any] = None,
        requires_clarification: bool = False
    ) -> Dict[str, Any]:
        """Generate a response to user query"""
        try:
            if requires_clarification:
                return await self._generate_clarifying_questions(user_query)
            
            # Check if query is off-topic
            if await self._is_off_topic(user_query):
                return await self._generate_redirect_response(user_query)
            
            # Synthesize information
            synthesized_info = await self._synthesize_information(
                user_query, retrieved_info or [], web_info or []
            )
            
            # Generate main response
            response = await self._generate_main_response(
                user_query, synthesized_info, memory_context or {}
            )
            
            return {
                'response': response,
                'sources': self._extract_sources(retrieved_info, web_info),
                'confidence': await self._calculate_confidence(response, retrieved_info),
                'timestamp': datetime.now().isoformat(),
                'requires_follow_up': await self._needs_follow_up(user_query, response)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': "I apologize, but I'm having trouble processing your request right now. Could you please rephrase your question?",
                'sources': [],
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat(),
                'requires_follow_up': False
            }
    
    async def _generate_clarifying_questions(self, user_query: str) -> Dict[str, Any]:
        """Generate clarifying questions"""
        try:
            prompt = self.clarification_prompt.format_messages(
                persona_name=self.persona.persona.name,
                persona_title=self.persona.persona.title,
                user_query=user_query
            )
            
            response = await self.primary_llm.agenerate([prompt])
            questions = response.generations[0][0].text.strip()
            
            return {
                'response': questions,
                'type': 'clarification',
                'sources': [],
                'confidence': 0.9,
                'timestamp': datetime.now().isoformat(),
                'requires_follow_up': True
            }
            
        except Exception as e:
            logger.error(f"Error generating clarifying questions: {e}")
            return {
                'response': "Could you tell me more about what you're looking for?",
                'type': 'clarification',
                'sources': [],
                'confidence': 0.5,
                'timestamp': datetime.now().isoformat(),
                'requires_follow_up': True
            }
    
    async def _is_off_topic(self, user_query: str) -> bool:
        """Check if query is off-topic for GenAI"""
        genai_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'deep learning',
            'llm', 'large language model', 'gpt', 'claude', 'gemini', 'llama',
            'generative', 'transformer', 'neural network', 'nlp', 'computer vision',
            'rag', 'retrieval augmented generation', 'vector database', 'embedding',
            'fine-tuning', 'prompt engineering', 'langchain', 'langgraph',
            'multimodal', 'diffusion', 'stable diffusion', 'midjourney', 'dall-e'
        ]
        
        query_lower = user_query.lower()
        return not any(keyword in query_lower for keyword in genai_keywords)
    
    async def _generate_redirect_response(self, user_query: str) -> Dict[str, Any]:
        """Generate redirect response for off-topic queries"""
        try:
            prompt = self.redirect_prompt.format_messages(
                persona_name=self.persona.persona.name,
                persona_title=self.persona.persona.title,
                user_query=user_query
            )
            
            response = await self.primary_llm.agenerate([prompt])
            redirect_text = response.generations[0][0].text.strip()
            
            return {
                'response': redirect_text,
                'type': 'redirect',
                'sources': [],
                'confidence': 0.8,
                'timestamp': datetime.now().isoformat(),
                'requires_follow_up': True
            }
            
        except Exception as e:
            logger.error(f"Error generating redirect response: {e}")
            return {
                'response': self.persona.get_redirect_message("that topic"),
                'type': 'redirect',
                'sources': [],
                'confidence': 0.7,
                'timestamp': datetime.now().isoformat(),
                'requires_follow_up': True
            }
    
    async def _synthesize_information(
        self,
        user_query: str,
        retrieved_info: List[Dict[str, Any]],
        web_info: List[Dict[str, Any]]
    ) -> str:
        """Synthesize information from multiple sources"""
        try:
            if not retrieved_info and not web_info:
                return "No specific information found."
            
            # Format retrieved documents
            retrieved_docs = "\n\n".join([
                f"Source: {doc.get('source', 'Unknown')}\nContent: {doc.get('content', '')[:500]}"
                for doc in retrieved_info[:3]  # Limit to top 3
            ])
            
            # Format web information
            web_formatted = "\n\n".join([
                f"Source: {info.get('source', 'Unknown')}\nTitle: {info.get('title', '')}\nContent: {info.get('content', '')[:300]}"
                for info in web_info[:3]  # Limit to top 3
            ])
            
            prompt = self.synthesis_prompt.format_messages(
                retrieved_docs=retrieved_docs,
                web_info=web_formatted,
                user_query=user_query
            )
            
            response = await self.secondary_llm.agenerate([prompt])
            synthesized = response.generations[0][0].text.strip()
            
            return synthesized
            
        except Exception as e:
            logger.error(f"Error synthesizing information: {e}")
            return "Information synthesis unavailable."
    
    async def _generate_main_response(
        self,
        user_query: str,
        synthesized_info: str,
        memory_context: Dict[str, Any]
    ) -> str:
        """Generate the main response"""
        try:
            prompt = self.response_prompt.format_messages(
                persona_name=self.persona.persona.name,
                persona_title=self.persona.persona.title,
                persona_background=self.persona.persona.background,
                persona_expertise=", ".join(self.persona.persona.expertise_areas),
                persona_personality=", ".join(self.persona.persona.personality_traits),
                persona_interaction_style=self.persona.persona.interaction_style,
                retrieved_info=synthesized_info,
                web_info="",  # Already included in synthesized_info
                memory_context=json.dumps(memory_context, indent=2),
                user_query=user_query
            )
            
            response = await self.primary_llm.agenerate([prompt])
            main_response = response.generations[0][0].text.strip()
            
            return main_response
            
        except Exception as e:
            logger.error(f"Error generating main response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Could you please try rephrasing your question?"
    
    def _extract_sources(
        self,
        retrieved_info: List[Dict[str, Any]],
        web_info: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Extract and format sources"""
        sources = []
        
        for doc in retrieved_info:
            sources.append({
                'title': doc.get('title', 'Unknown'),
                'url': doc.get('url', ''),
                'source': doc.get('source', 'Unknown')
            })
        
        for info in web_info:
            sources.append({
                'title': info.get('title', 'Unknown'),
                'url': info.get('url', ''),
                'source': info.get('source', 'Unknown')
            })
        
        return sources
    
    async def _calculate_confidence(
        self,
        response: str,
        retrieved_info: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for the response"""
        try:
            # Simple confidence calculation based on available information
            base_confidence = 0.5
            
            if retrieved_info:
                base_confidence += 0.3
            
            if len(response) > 100:  # Substantial response
                base_confidence += 0.1
            
            if any(keyword in response.lower() for keyword in ['according to', 'research shows', 'studies indicate']):
                base_confidence += 0.1
            
            return min(base_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    async def _needs_follow_up(self, user_query: str, response: str) -> bool:
        """Determine if follow-up is needed"""
        follow_up_indicators = [
            'would you like to know more',
            'do you have questions',
            'feel free to ask',
            'let me know if you need'
        ]
        
        return any(indicator in response.lower() for indicator in follow_up_indicators)
