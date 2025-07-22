# Knowledge Base Service

Independent microservice for document storage, vector search, and RAG operations.

## Features

- Document ingestion and chunking
- Vector similarity search
- Hybrid retrieval (Vector + BM25)
- Caching for improved performance
- RESTful API with OpenAPI documentation

## API Endpoints

- `POST /api/v1/documents` - Create new document
- `GET /api/v1/search` - Search documents
- `GET /api/v1/documents/{id}` - Get specific document
- `DELETE /api/v1/documents/{id}` - Delete document
- `GET /health` - Health check

## Configuration

Set environment variables:

```bash
OPENAI_API_KEY=your_key
VECTOR_STORE_TYPE=pinecone  # pinecone, weaviate, chroma
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=your_index
REDIS_URL=redis://localhost:6379
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run service
python -m src.main

# Run tests
pytest tests/
```

## Docker

```bash
# Build image
docker build -t knowledge-base-service .

# Run container
docker run -p 8002:8002 knowledge-base-service
```

## Health Check

```bash
curl http://localhost:8002/health
```