from pydantic import BaseModel
from typing import List, Dict, Any

class Document(BaseModel):
    doc_id: str | None
    content: str
    metadata: Dict[str, Any] = {}


class IndexRequest(BaseModel):
    index_name: str


class RetrieverRequest(BaseModel):
    retriever_name: str
    index_name: str


class QueryRequest(BaseModel):
    retriever_name: str
    query: str