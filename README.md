# AStarBot - Personal AI Assistant

A comprehensive README file for **AStarBot**, Avrodeep Pal's personal AI assistant powered by Retrieval-Augmented Generation (RAG) technology.

## 🤖 Overview

**AStarBot** is an intelligent personal AI assistant designed to answer questions about Avrodeep Pal's professional background, skills, projects, and experience. Built with Python, FastAPI, and RAG technology, it combines the power of large language models with personalized knowledge retrieval to provide accurate, contextual responses about Avrodeep's portfolio and capabilities.

The system features both a command-line interface for development and testing, and a RESTful API backend that can be integrated with frontend applications for seamless user interactions.

## ✨ Key Features

- **🧠 RAG-Powered Intelligence**: Utilizes Retrieval-Augmented Generation for accurate, context-aware responses
- **💾 Redis Memory Support**: Persistent conversation memory across sessions with Redis Cloud integration
- **🌐 FastAPI Backend**: High-performance REST API with automatic documentation
- **💬 Interactive CLI**: Command-line interface for direct testing and interaction
- **📊 Session Management**: Multiple conversation sessions with memory isolation
- **🔍 Vector Search**: Pinecone integration for semantic similarity search
- **🎯 Portfolio-Focused**: Specialized knowledge base about Avrodeep's skills, projects, and experience
- **🚀 Production-Ready**: Modular architecture with proper error handling and logging

## 🏗️ Architecture

### Project Structure
```
astarbot/
├── data/                    # Knowledge base and embeddings
│   ├── self_data.json      # Structured personal information
│   └── embed_data.py       # Data processing and embedding utilities
├── rag/                     # RAG system components
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── prompt.py           # Prompt templates and formatting
│   └── rag_chain.py        # Core RAG implementation
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── helpers.py          # General helper functions
│   └── memory.py           # Redis memory management
├── main.py                 # FastAPI server application
├── app.py                  # CLI interface
├── requirements.txt        # Python dependencies
├── structure.txt          # Project structure documentation
└── README.md              # This file
```

### Technology Stack

**Backend Framework**
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for production deployment

**AI/ML Components**
- **OpenAI**: GPT models for text generation
- **Sentence Transformers**: Text embeddings for semantic search
- **Hugging Face Hub**: Model management and deployment

**LangChain Ecosystem**
- **LangChain**: Framework for LLM application development
- **LangChain Community**: Additional integrations and tools
- **LangChain Core**: Core functionality and abstractions

**Vector Database & Memory**
- **Pinecone**: Cloud-based vector database for similarity search
- **Redis**: In-memory data store for conversation history

**Utilities**
- **Python-dotenv**: Environment variable management
- **Pydantic**: Data validation and serialization

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key
- Redis Cloud instance (optional, for memory features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AvrodeepPal/AStarBot.git
   cd AStarBot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   REDIS_URL=your_redis_url  # Optional
   REDIS_PASSWORD=your_redis_password  # Optional
   ```

4. **Initialize the vector database**
   ```bash
   python data/embed_data.py
   ```

### Usage

#### 🖥️ Command Line Interface

For interactive testing and development:

```bash
python app.py
```

**Available CLI Commands:**
- `!help` - Show help message and available commands
- `!session` - Show current session information
- `!clear` - Clear current session memory
- `!new` - Start a new session
- `!memory` - Show memory information
- `!config` - Show configuration status
- `!quit` - Exit the application

**Example Interaction:**
```
🤖 AStarBot CLI - Avrodeep Pal's AI Assistant
🧠 Enhanced with Redis Memory & Session Support

Enter session ID (press Enter for 'guest'): demo

[demo] 🧑 You: What are your main programming languages?
[demo] 🤖 AStarBot: I'm proficient in several programming languages including C, C++, Java, SQL, and even 8085 Assembly. I work comfortably across the stack with modern web technologies like HTML, CSS, JavaScript, React, and Next.js, along with backend frameworks like FastAPI. I also have experience with Python for AI/ML projects.
```

#### 🌐 REST API Server

For production deployment and frontend integration:

```bash
python main.py
```

The API will be available at `http://localhost:8000` with automatic documentation at `http://localhost:8000/docs`.

**API Endpoints:**

- `GET /` - Health check and service information
- `GET /health` - Detailed health status
- `POST /chat` - Main chat endpoint (structured)
- `POST /chat-simple` - Simplified chat endpoint

**Example API Usage:**
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"question": "Tell me about your recent projects"}'
```

**Response:**
```json
{
  "answer": "I've been working on several exciting projects recently. 'Let's Connect!' is an AI-powered platform I built to streamline campus recruitment outreach using LLMs and Streamlit, achieving 3x faster outreach. I also developed a stock price prediction model comparing ReLU and GELU activations in LSTM networks, and created Star Emporium, a Java GUI e-book library focused on accessibility and eco-efficiency.",
  "status": "success"
}
```

## 📚 Knowledge Base

AStarBot's knowledge base contains comprehensive information about Avrodeep Pal, including:

### 🎓 Education & Achievements
- Current MCA student at Jadavpur University (CGPA: 8.33)
- B.Sc. (Hons.) Computer Science from WBSU (CGPA: 9.91, Rank 1)
- WB-JECA 2024 Rank 2

### 💻 Technical Skills
- **Programming Languages**: C, C++, Java, Python, SQL, Assembly
- **Web Technologies**: HTML, CSS, JavaScript, React, Next.js, Tailwind CSS
- **Backend**: FastAPI, Node.js
- **Databases**: SQL, Supabase
- **Tools**: Git, Ubuntu, Various AI/ML frameworks

### 🚀 Featured Projects
- **Let's Connect!**: AI-powered recruitment outreach platform
- **Stock Price Prediction**: LSTM-based model with activation function comparison
- **Star Emporium**: Java GUI e-book library system
- **AStarBot**: This very AI assistant you're interacting with!

### 👨💼 Professional Interests
- Machine Learning and LLM pipelines
- Full-stack web development
- AI-powered applications
- Open-source contributions
- Personalized and adaptive systems

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes |
| `PINECONE_API_KEY` | Pinecone API key for vector database | Yes |
| `PINECONE_ENVIRONMENT` | Pinecone environment (e.g., 'us-west1-gcp') | Yes |
| `REDIS_URL` | Redis connection URL | No |
| `REDIS_PASSWORD` | Redis password | No |
| `ENABLE_MEMORY` | Enable/disable memory features | No |

### Customization

To adapt AStarBot for your own use:

1. **Update Knowledge Base**: Modify `data/self_data.json` with your information
2. **Adjust Prompts**: Edit `rag/prompt.py` to change response style
3. **Configure Models**: Update `rag/config.py` for different AI models
4. **Extend Features**: Add new modules in the `utils/` directory

## 🔧 Development

### Adding New Features

1. **Create new modules** in appropriate directories
2. **Update configuration** in `rag/config.py`
3. **Add API endpoints** in `main.py`
4. **Include CLI commands** in `app.py`
5. **Update documentation** and tests

### Testing

```bash
# Test the CLI interface
python app.py

# Test API endpoints
python main.py
# Visit http://localhost:8000/docs for interactive testing
```

### Deployment

#### Using Uvicorn (Production)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Using Docker (Recommended)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

### Development Guidelines
1. Follow PEP 8 style guidelines
2. Add docstrings to new functions
3. Update tests for new features
4. Update documentation as needed

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact & Support

**Avrodeep Pal**
- 📧 Email: avrodeep.pal17@gmail.com
- 📱 Phone: +91-8583842681
- 💼 LinkedIn: [linkedin.com/in/avrodeeppal](https://linkedin.com/in/avrodeeppal)
- 🐙 GitHub: [github.com/AvrodeepPal](https://github.com/AvrodeepPal)
- 📄 Resume: [tinyurl.com/AvrodeepPal](https://tinyurl.com/AvrodeepPal)

## 🙏 Acknowledgments

- **OpenAI** for powerful language models
- **LangChain** for excellent RAG framework
- **FastAPI** for the modern web framework
- **Pinecone** for scalable vector database
- **Redis** for reliable memory management
- The open-source community for inspiration and tools

**Built with ❤️ by Avrodeep Pal | Making AI personal, one conversation at a time**