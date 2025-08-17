# 🔹 Bizz | Docs - AI Document Analysis System

**Bizz | Docs** je AI-powered sistem za analizu dokumenata koji omogućava zaposlenima da jednostavno ubace PDF-ove i dobiju kompletan opis ili izdvoje ključne informacije.

## 🚀 Features

- **📄 Multi-format Support**: PDF, Word, TXT, Markdown
- **🤖 AI Analysis**: ChatGPT API powered analysis
- **🔍 Smart Retrieval**: Vector-based document search
- **💬 Interactive Chat**: Natural language queries
- **📊 Multiple Analysis Types**: Summary, key points, specific info extraction
- **🌐 Web Interface**: Streamlit frontend + FastAPI backend
- **💾 Persistent Storage**: ChromaDB vector database

## 🛠️ Tech Stack

- **Frontend**: Streamlit (rapid prototyping)
- **Backend**: FastAPI (Python web framework)
- **AI/LLM**: OpenAI GPT-4o mini
- **Embeddings**: HuggingFace sentence-transformers
- **Vector DB**: ChromaDB (local, free)
- **Orchestration**: LlamaIndex
- **Document Parsing**: PyPDF2, python-docx

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Internet connection (for API calls)

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo>
cd bizz-docs
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 4. Run Streamlit App

```bash
streamlit run app.py
```

### 5. Run FastAPI Backend (Optional)

```bash
python api.py
```

## 📖 Usage Guide

### Streamlit App (Recommended for Start)

1. **Initialize RAG System**: Click "🚀 Initialize RAG System" in sidebar
2. **Upload Documents**: Drag & drop PDF, Word, or text files
3. **Process Documents**: Click "📚 Process Documents"
4. **Ask Questions**: Use chat interface to query your documents
5. **Choose Analysis Type**:
   - 💬 **General Question**: Ask anything about your documents
   - 📋 **Summary**: Get comprehensive document summary
   - 🎯 **Key Points**: Extract main insights and findings
   - 🔍 **Extract Info**: Find specific information

### API Endpoints

- `POST /initialize` - Initialize RAG system
- `POST /upload` - Upload and process document
- `POST /query` - Query documents with AI
- `GET /status` - Get system status
- `GET /documents` - Get document information
- `DELETE /documents` - Clear all documents

## 🔧 Configuration

### Model Selection

- **LLM**: OpenAI GPT-4o mini (default)
- **Embeddings**: HuggingFace MiniLM (free, local)
- **Vector DB**: ChromaDB (local, persistent)

### Environment Variables

```bash
OPENAI_API_KEY=your-openai-api-key
```

## 📁 Project Structure

```
bizz-docs/
├── app.py                 # Main Streamlit application
├── api.py                 # FastAPI backend
├── document_parser.py     # Document parsing logic
├── rag_system.py         # RAG system core
├── chat_interface.py     # Chat UI components
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── chroma_db/           # Vector database (auto-created)
```

## 🎯 Use Cases

### For Business Users

1. **📋 Document Summaries**: Quick overview of long reports
2. **🎯 Key Insights**: Extract main points from contracts
3. **🔍 Information Retrieval**: Find specific data in documents
4. **📊 Content Analysis**: Understand document themes and topics

### For Teams

1. **📚 Knowledge Base**: Centralized document analysis
2. **🤝 Collaboration**: Share insights from documents
3. **⚡ Quick Research**: Fast answers from company documents
4. **📈 Decision Support**: Data-driven insights from reports

## 🚀 Deployment Options

### Local Development

```bash
streamlit run app.py
```

### Production Deployment

1. **Streamlit Cloud**: Free hosting for Streamlit apps
2. **Vercel**: Deploy FastAPI backend
3. **Docker**: Containerized deployment
4. **Cloud Platforms**: AWS, GCP, Azure

## 🔒 Security Considerations

- API keys stored in environment variables
- File upload validation
- Temporary file cleanup
- CORS configuration for production

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Error**: Check API key and billing
2. **Document Parsing Error**: Verify file format support
3. **Memory Issues**: Large documents may require chunking
4. **Vector DB Error**: Check disk space and permissions

### Debug Mode

```bash
export LOG_LEVEL=DEBUG
streamlit run app.py
```

## 📈 Performance Tips

1. **Document Size**: Keep documents under 50MB
2. **Chunk Size**: Optimize for your use case (default: 512 tokens)
3. **Batch Processing**: Upload multiple documents at once
4. **Caching**: Streamlit caches results automatically

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced document types (Excel, PowerPoint)
- [ ] Team collaboration features
- [ ] Advanced analytics dashboard
- [ ] Integration with business tools
- [ ] Custom model fine-tuning

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: This README
- **Community**: GitHub Discussions

---

**🔹 Bizz | Docs** - Making document analysis simple and intelligent! 🚀
