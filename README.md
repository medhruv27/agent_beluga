# GenAI Research Assistant - Sophisticated Human-like Chatbot

A sophisticated human-like chatbot that adopts a unique identity and specializes in Generative AI. Built with a multi-level agent architecture for reliable, contextually aware responses.

## 🤖 Meet beluga

**beluga** is a Generative AI Research Analyst with over 5 years of experience exploring the fascinating world of AI. She's passionate about helping people navigate the rapidly evolving field of Generative AI, from the latest models and frameworks to research trends and best practices.

### Key Features
- **Human-like Identity**: Sophia presents herself as a real person with personal details and backstory
- **Multi-Agent Architecture**: 5 specialized agents working together for optimal responses
- **Cross-Questioning**: Asks clarifying questions to ensure personalized, relevant answers
- **Memory System**: Both short-term and long-term memory for personalization
- **Cost Optimization**: Smart model selection and prompt compression
- **Comprehensive Evaluation**: Detailed metrics and CSV reporting

## 🏗️ Architecture

### Multi-Agent System

1. **Retrieval Agent** - Queries Qdrant vector database for embeddings and domain-specific information
2. **Web-Scraper Agent** - Fetches latest updates, papers, and market reports using Playwright/BeautifulSoup
3. **Reasoning Agent** - Uses LLMs to synthesize retrieved and scraped data
4. **Memory Agent** - Maintains short-term (Redis) and long-term (Qdrant) memory
5. **Evaluation Agent** - Validates responses against hallucination and relevance using Ragas/DeepEval

### Tech Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Vector Database**: Qdrant (Required)
- **Agent Orchestration**: LangChain, LangGraph
- **LLMs**: OpenAI GPT, Google Gemini, Anthropic Claude
- **Evaluation Tools**: Ragas, DeepEval, TruLens
- **Memory**: Redis + Qdrant
- **Web Scraping**: Playwright, BeautifulSoup

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker (for Qdrant and Redis)
- OpenAI API key
- Google API key (optional)
- Anthropic API key (optional)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Start required services**
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start Redis
docker run -p 6379:6379 redis:alpine
```

6. **Run the application**
```bash
# Start FastAPI backend
python main.py

# In another terminal, start Streamlit frontend
streamlit run streamlit_app.py
```

## 📋 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here

# Redis Configuration
REDIS_URL=redis://localhost:6379

# LangSmith (for evaluation)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=genai-chatbot

# Application Settings
APP_NAME=GenAI Research Assistant
DEBUG=True
LOG_LEVEL=INFO
```

### Model Configuration

The system supports multiple LLMs with automatic cost optimization:

- **Primary LLM**: GPT-4 (for complex queries)
- **Secondary LLM**: GPT-3.5-turbo (for simple tasks)
- **Embedding Model**: text-embedding-ada-002

## 🎯 Usage

### Web Interface

1. Open your browser to `http://localhost:8501`
2. Start chatting with Sophia about Generative AI topics
3. She will ask clarifying questions to better understand your needs
4. View evaluation metrics and sources in the sidebar

### API Usage

```python
import requests

# Send a chat message
response = requests.post("http://localhost:8000/chat", json={
    "message": "What are the latest trends in large language models?",
    "session_id": "your-session-id"
})

print(response.json())
```

### Example Queries

- "What are the latest trends in large language models?"
- "Compare LangChain vs LangGraph for building AI agents"
- "How do I implement RAG with vector databases?"
- "What's the best approach for fine-tuning LLMs?"
- "Explain multimodal AI applications"

## 📊 Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Core Metrics
- **Hallucination Rate** (25%): Minimal hallucinations using evaluation tools
- **Contextual Relevance** (25%): Responses align with user queries
- **Multi-Agent Architecture** (20%): Proper implementation of all 5 agents
- **Human-Like Behavior** (15%): Natural engagement with clarifying questions
- **Memory Utilization** (15%): Effective use of short/long-term memory

### Custom Evaluation Matrix
- Domain-specificity
- Technical accuracy
- Practical utility
- Response continuity

### CSV Reports

The system automatically generates CSV reports with:
- Individual query evaluations
- Summary statistics
- Evaluation matrix
- Cost analysis

## 🔧 Development

### Project Structure

```
chatbot/
├── agents/                 # Multi-agent system
│   ├── retrieval_agent.py
│   ├── webscraper_agent.py
│   ├── reasoning_agent.py
│   ├── memory_agent.py
│   └── evaluation_agent.py
├── main.py                # FastAPI backend
├── streamlit_app.py       # Streamlit frontend
├── persona.py             # Human persona system
├── config.py              # Configuration settings
├── optimization.py        # Cost optimization
├── evaluation_metrics.py  # Evaluation and reporting
└── requirements.txt       # Dependencies
```

### Adding New Agents

1. Create a new agent class in `agents/`
2. Implement required methods
3. Add to agent orchestration in `main.py`
4. Update evaluation metrics

### Customizing Persona

Edit `persona.py` to modify Sophia's:
- Background and story
- Expertise areas
- Interaction style
- Clarifying questions

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d
```

### Cloud Deployment

The application is designed to be deployed on cloud platforms:

- **Backend**: Deploy FastAPI to AWS/GCP/Azure
- **Frontend**: Deploy Streamlit to Streamlit Cloud
- **Database**: Use managed Qdrant and Redis services
- **Monitoring**: Integrate with LangSmith for evaluation

## 📈 Performance Optimization

### Cost Optimization Features

- **Smart Model Selection**: Uses cheaper models for simple queries
- **Prompt Compression**: Reduces token usage while preserving context
- **Conversation Summarization**: Maintains context with fewer tokens
- **Hybrid Retrieval**: Combines vector and keyword search

### Performance Monitoring

- Real-time cost tracking
- Token usage monitoring
- Response time metrics
- Quality score tracking

## 🧪 Testing

### Running Tests

```bash
# Run evaluation tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=. tests/
```

### Evaluation Queries

The system includes 50+ test queries covering:
- Basic GenAI concepts
- Technical implementations
- Framework comparisons
- Research trends
- Best practices

## 📚 Resources

### Helpful Links
- [LangChain Documentation](https://python.langchain.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Research Sources
- [LangChain Blog](https://blog.langchain.com/)
- [LlamaIndex Blog](https://www.llamaindex.ai/blog)
- [Hugging Face Blog](https://huggingface.co/blog)
- [OpenAI Blog](https://openai.com/blog)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the evaluation metrics for troubleshooting

## 🎯 Roadmap

- [ ] Integration with more LLM providers
- [ ] Advanced web scraping capabilities
- [ ] Real-time collaboration features
- [ ] Mobile app interface
- [ ] Advanced analytics dashboard

---

**Built with ❤️ for the Generative AI community**
