# LLM-Powered-Retrieval-System
# Vector Database System

A robust and flexible vector database system built with FastAPI and LlamaIndex, featuring advanced querying capabilities, metadata filtering, and hybrid search functionality.

## Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Creating an Index](#creating-an-index)
  - [Adding Vectors](#adding-vectors)
  - [Querying](#querying)
  - [Metadata Filtering](#metadata-filtering)
  - [Delete Operations](#delete-operations)
- [API Reference](#api-reference)
- [Advanced Features](#advanced-features)
- [Error Handling](#error-handling)

## Features

- **Vector Storage & Retrieval**: Custom vector store implementation with efficient storage and retrieval mechanisms
- **Advanced Metadata Filtering**: Support for complex nested filters with AND/OR conditions
- **Hybrid Search**: Combines semantic similarity with keyword matching for improved search accuracy
- **CRUD Operations**: Complete Create, Read, Update, Delete operations for vectors and indices
- **Persistent Storage**: Automatic persistence of vectors and metadata to disk
- **Query Engine Management**: Create and manage multiple query engines with different configurations
- **Token-Based Access**: Secure access to indices using unique tokens
- **Enhanced Error Handling**: Comprehensive error tracking and reporting

## System Architecture

The system consists of two main components:

1. **CustomVectorStore**: A custom implementation of LlamaIndex's VectorStore
   - Handles vector storage and retrieval
   - Implements metadata filtering
   - Manages persistence to disk

2. **FastAPI Application**: REST API interface
   - Manages indexes and query engines
   - Handles CRUD operations
   - Implements complex querying logic

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Environment setup
cp .env.example .env
# Add your OpenAI API key to .env
```

Required Environment Variables:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Creating an Index

```python
# Create a new index
POST /create_index/
{
    "index_name": "my_index"
}
```

Response:
```json
{
    "message": "Index my_index created successfully.",
    "token": "unique-token-id"
}
```

### Adding Vectors

```python
# Add vectors to an index
POST /add_vector/
{
    "token": "your-token",
    "content": ["Your text content here"],
    "metadata": {
        "category": "example",
        "tags": ["tag1", "tag2"]
    }
}
```

### Querying

Basic Query:
```python
POST /query_index/
{
    "token": "your-token",
    "query": "Your query here",
    "query_engine_id": "optional-engine-id"
}
```

Advanced Query with Filters:
```python
POST /query_index/
{
    "token": "your-token",
    "query": "Your query here",
    "filter_groups": [
        {
            "condition": "AND",
            "filters": [
                {
                    "key": "category",
                    "value": "example",
                    "operator": "=="
                },
                {
                    "key": "date",
                    "value": "2024-01-01",
                    "operator": ">="
                }
            ]
        }
    ],
    "top_level_condition": "AND"
}
```

### Metadata Filtering

The system supports complex metadata filtering with various operators:

- `==`: Equal to
- `!=`: Not equal to
- `>`: Greater than
- `>=`: Greater than or equal to
- `<`: Less than
- `<=`: Less than or equal to
- `contains`: Contains value

Nested filters example:
```python
{
    "filter_groups": [
        {
            "condition": "OR",
            "filters": [
                {
                    "key": "category",
                    "value": "tech",
                    "operator": "=="
                },
                {
                    "key": "tags",
                    "value": "AI",
                    "operator": "contains"
                }
            ]
        },
        {
            "condition": "AND",
            "filters": [
                {
                    "key": "date",
                    "value": "2024-01-01",
                    "operator": ">="
                }
            ]
        }
    ],
    "top_level_condition": "AND"
}
```

### Delete Operations

Delete Vectors:
```python
POST /delete_nodes/
{
    "token": "your-token",
    "metadata_filters": [
        {
            "key": "category",
            "value": "example",
            "operator": "=="
        }
    ]
}
```

Delete Index:
```python
DELETE /index/
{
    "token": "your-token"
}
```

## Advanced Features

### Hybrid Search

The system implements a hybrid search mechanism that combines:
- Semantic similarity using embeddings
- Keyword-based matching
- Metadata filtering

The hybrid search provides better search results by:
- Considering semantic meaning of the query
- Matching specific keywords
- Applying metadata filters
- Using customizable weights for different search components

### Query Engine Management

Create custom query engines with specific configurations:
```python
POST /create_query_engine/
{
    "token": "your-token",
    "query_engine_name": "custom_engine",
    "metadata_filters": [...],
    "filter_groups": [...]
}
```

## Error Handling

The system provides detailed error messages and status codes:
- 404: Resource not found (invalid token, missing index)
- 400: Bad request (invalid parameters)
- 500: Internal server error (processing errors)

Error responses include:
- Detailed error message
- Stack trace (in development)
- Affected components
- Partial results (when applicable)

## Best Practices

1. **Index Management**
   - Create separate indices for different data types
   - Use meaningful index names
   - Regularly backup index data

2. **Metadata Design**
   - Use consistent metadata structure
   - Include relevant search fields
   - Consider query patterns when designing metadata

3. **Query Optimization**
   - Use appropriate filter combinations
   - Leverage hybrid search capabilities
   - Monitor query performance

4. **Error Handling**
   - Implement proper error handling in your client
   - Log and monitor error responses
   - Handle token expiration and renewal

## Limitations

- Maximum vector dimension is determined by the embedding model
- Query response time may increase with larger indices
- Metadata filters should be used judiciously to maintain performance
- Token storage requires proper security measures

## Troubleshooting

Common issues and solutions:

1. **Query Returns No Results**
   - Check metadata filters
   - Verify token validity
   - Ensure index contains data

2. **Poor Search Quality**
   - Adjust hybrid search weights
   - Review metadata structure
   - Consider using different filter combinations

3. **Performance Issues**
   - Optimize index size
   - Review filter complexity
   - Check system resources



## License

