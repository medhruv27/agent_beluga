"""
Streamlit frontend for the GenAI Chatbot
"""
import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, Any, List
import uuid

# Configure Streamlit page
st.set_page_config(
    page_title="GenAI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .persona-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .message-user {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .message-bot {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    .source-item {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'persona_info' not in st.session_state:
        st.session_state.persona_info = None
    
    if 'evaluation_summary' not in st.session_state:
        st.session_state.evaluation_summary = None


def get_persona_info() -> Dict[str, Any]:
    """Get persona information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/persona")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching persona info: {e}")
        return None


def send_chat_message(message: str, session_id: str) -> Dict[str, Any]:
    """Send chat message to API"""
    try:
        payload = {
            "message": message,
            "session_id": session_id
        }
        response = requests.post(f"{API_BASE_URL}/chat", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error sending message: {e}")
        return None


def get_evaluation_summary() -> Dict[str, Any]:
    """Get evaluation summary from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/evaluation/summary")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching evaluation summary: {e}")
        return None


def display_persona_info(persona_info: Dict[str, Any]):
    """Display persona information"""
    if not persona_info:
        return
    
    st.markdown(f"""
    <div class="persona-card">
        <h3>👋 Meet {persona_info['name']}</h3>
        <p><strong>Title:</strong> {persona_info['title']}</p>
        <p><strong>Background:</strong> {persona_info['background']}</p>
        <p><strong>Expertise:</strong> {', '.join(persona_info['expertise_areas'])}</p>
    </div>
    """, unsafe_allow_html=True)


def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message"""
    message_class = "message-user" if is_user else "message-bot"
    
    st.markdown(f"""
    <div class="{message_class}">
        <strong>{'You' if is_user else 'Sophia'}:</strong><br>
        {message['content']}
    </div>
    """, unsafe_allow_html=True)
    
    # Display sources if available
    if not is_user and message.get('sources'):
        with st.expander("📚 Sources"):
            for source in message['sources']:
                st.markdown(f"""
                <div class="source-item">
                    <strong>{source.get('title', 'Unknown')}</strong><br>
                    <em>Source:</em> {source.get('source', 'Unknown')}<br>
                    <em>URL:</em> {source.get('url', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
    
    # Display confidence and evaluation scores
    if not is_user:
        col1, col2 = st.columns(2)
        with col1:
            if message.get('confidence'):
                st.metric("Confidence", f"{message['confidence']:.2f}")
        with col2:
            if message.get('evaluation_score'):
                st.metric("Quality Score", f"{message['evaluation_score']:.2f}")


def display_evaluation_metrics(summary: Dict[str, Any]):
    """Display evaluation metrics"""
    if not summary:
        return
    
    st.markdown("### 📊 Evaluation Metrics")
    
    # Create columns for metrics
    cols = st.columns(4)
    
    metrics = ['overall', 'hallucination', 'relevance', 'context_precision']
    metric_names = ['Overall Score', 'Hallucination Rate', 'Relevance', 'Context Precision']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        with cols[i]:
            if metric in summary:
                value = summary[metric]['mean']
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{name}</h4>
                    <h2>{value:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)


def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 GenAI Research Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 About This Assistant")
        
        # Get and display persona info
        if not st.session_state.persona_info:
            with st.spinner("Loading persona information..."):
                st.session_state.persona_info = get_persona_info()
        
        if st.session_state.persona_info:
            display_persona_info(st.session_state.persona_info)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Ask about **latest AI models** and frameworks
        - Inquire about **best practices** in GenAI
        - Request **comparisons** between different tools
        - Ask for **tutorials** and implementation guides
        - Discuss **research trends** and papers
        """)
        
        st.markdown("### 📈 Evaluation Metrics")
        if st.button("Refresh Metrics"):
            st.session_state.evaluation_summary = get_evaluation_summary()
        
        if st.session_state.evaluation_summary:
            display_evaluation_metrics(st.session_state.evaluation_summary)
    
    # Main chat interface
    st.markdown("### 💬 Chat with Sophia")
    
    # Chat input
    user_input = st.text_input(
        "Ask me anything about Generative AI:",
        placeholder="e.g., What are the latest trends in large language models?",
        key="user_input"
    )
    
    # Send button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        send_button = st.button("Send", type="primary")
    with col2:
        clear_button = st.button("Clear Chat")
    
    # Handle clear chat
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    # Handle send message
    if send_button and user_input:
        # Add user message to history
        user_message = {
            'content': user_input,
            'timestamp': datetime.now().isoformat(),
            'type': 'user'
        }
        st.session_state.chat_history.append(user_message)
        
        # Send to API and get response
        with st.spinner("Sophia is thinking..."):
            response = send_chat_message(user_input, st.session_state.session_id)
        
        if response:
            # Add bot response to history
            bot_message = {
                'content': response['response'],
                'timestamp': response['timestamp'],
                'type': 'bot',
                'sources': response.get('sources', []),
                'confidence': response.get('confidence', 0),
                'evaluation_score': response.get('evaluation_score')
            }
            st.session_state.chat_history.append(bot_message)
        
        # Clear input
        st.session_state.user_input = ""
        st.rerun()
    
    # Display chat history
    st.markdown("### 📝 Conversation History")
    
    if not st.session_state.chat_history:
        st.info("👋 Hi! I'm Sophia, your Generative AI Research Assistant. Ask me anything about AI models, frameworks, research trends, or best practices!")
    else:
        for message in st.session_state.chat_history:
            is_user = message['type'] == 'user'
            display_chat_message(message, is_user)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Powered by Multi-Agent Architecture | Built with FastAPI & Streamlit</p>
        <p>Session ID: {}</p>
    </div>
    """.format(st.session_state.session_id), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
