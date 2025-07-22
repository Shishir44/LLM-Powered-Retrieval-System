# 🤖 LLM-Powered Retrieval System - Streamlit UI

A simple web-based user interface for testing all the functionality of the LLM-Powered Retrieval System microservices.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- The microservices should be running:
  - API Gateway: `http://localhost:8000`
  - Knowledge Base Service: `http://localhost:8002`
  - Conversation Service: `http://localhost:8001`

### Installation & Running

1. **Install dependencies:**
   ```bash
   pip install -r streamlit_requirements.txt
   ```

2. **Run the UI:**
   ```bash
   # Option 1: Using the run script
   python run_streamlit.py
   
   # Option 2: Direct streamlit command
   streamlit run streamlit_app.py --server.port 8501
   ```

3. **Access the UI:**
   Open your browser and go to: `http://localhost:8501`

## 🎯 Features

### 📚 Knowledge Base Management
- **Create Documents:** Add new documents with title, content, category, subcategory, and tags
- **Search Documents:** Search through the knowledge base with filtering options
- **View Results:** See search results with relevance scores and document details

### 💬 Chat Interface
- **Interactive Chat:** Send messages to the conversation service
- **Chat History:** Maintain conversation context across messages
- **Real-time Responses:** Get responses from the RAG pipeline

### 📊 System Statistics
- **Document Metrics:** Total documents, chunks, and averages
- **Category Breakdown:** Visual representation of document categories
- **Real-time Data:** Refresh statistics on demand

### 📋 Document Management
- **List All Documents:** Browse all documents with filtering
- **Document Details:** View complete document information
- **Category Filtering:** Filter by document categories

### 🔧 Service Health Monitoring
- **Real-time Status:** Check if all services are running
- **Health Dashboard:** Visual indicators for service availability
- **Error Handling:** Graceful handling of service outages

## 🎨 UI Components

### Main Tabs
1. **Knowledge Base:** Document creation and search functionality
2. **Chat:** Conversation interface with the RAG system
3. **Statistics:** System metrics and analytics
4. **Document List:** Browse and manage documents

### Sidebar
- Service health status indicators
- Refresh controls
- Quick service information

## 🔧 Configuration

### Service URLs
The UI is configured to connect to the following services:
```python
API_GATEWAY_URL = "http://localhost:8000"
KNOWLEDGE_BASE_URL = f"{API_GATEWAY_URL}/knowledge"
CONVERSATION_URL = f"{API_GATEWAY_URL}/conversation"
```

### Customization
To modify service endpoints, edit the configuration variables at the top of `streamlit_app.py`.

## 🧪 Testing Workflow

### 1. Test Knowledge Base
1. Go to the **Knowledge Base** tab
2. Create a test document:
   - Title: "Test Document"
   - Content: "This is a test document for the RAG system"
   - Category: "Technology"
   - Tags: "test, demo"
3. Click "Create Document"
4. Search for the document using keywords from the content
5. Verify the document appears in search results

### 2. Test Chat Interface
1. Go to the **Chat** tab
2. Send a message like "Hello, how can you help me?"
3. Verify you receive a response
4. Continue the conversation to test context management

### 3. Test Statistics
1. Go to the **Statistics** tab
2. Click "Refresh Statistics"
3. Verify document counts and category breakdown are displayed

### 4. Test Document List
1. Go to the **Document List** tab
2. Click "Load Documents"
3. Verify all documents are listed with correct information

## 🐛 Troubleshooting

### Common Issues

**Service Connection Errors:**
- Check if all microservices are running
- Verify service URLs in the sidebar status
- Ensure ports 8000, 8001, 8002 are accessible

**Missing Dependencies:**
```bash
pip install -r streamlit_requirements.txt
```

**Port Already in Use:**
```bash
# Use a different port
streamlit run streamlit_app.py --server.port 8502
```

### Service Health Check
The UI continuously monitors service health. If a service is down, you'll see:
- ❌ Red indicator in the sidebar
- Error messages when trying to use that service's features
- Graceful degradation of functionality

## 📝 API Integration

The UI integrates with the following API endpoints:

### Knowledge Base Service
- `POST /knowledge/documents` - Create document
- `GET /knowledge/search` - Search documents
- `GET /knowledge/documents` - List documents
- `GET /knowledge/stats` - Get statistics

### Conversation Service
- `POST /conversation/api/v1/chat` - Send chat message
- `GET /conversation/api/v1/conversations/{id}` - Get conversation

### API Gateway
- `GET /health` - Service health check

## 🔒 Security Notes

- The UI is configured for local development
- CORS is enabled for testing purposes
- For production deployment, configure proper authentication and CORS policies

## 🚀 Next Steps

1. **Enhanced Features:**
   - File upload for document creation
   - Conversation export functionality
   - Advanced search filters
   - Batch document operations

2. **UI Improvements:**
   - Dark mode support
   - Better error handling
   - Loading animations
   - Responsive design

3. **Production Setup:**
   - Environment configuration
   - Authentication integration
   - HTTPS support
   - Container deployment