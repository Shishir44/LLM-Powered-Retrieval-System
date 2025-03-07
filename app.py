from fastapi import FastAPI, HTTPException, responses
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Any, Dict, Union
import os
import json
import uuid
import hashlib
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor import KeywordNodePostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.postprocessor.types import BaseNodePostprocessor 
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition
)
from custom_vector_store import CustomVectorStore
# from custom_response_processor import NaturalLanguagePostProcessor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, NodeWithScore, NodeRelationship, RelatedNodeInfo, ObjectType, QueryBundle
import nest_asyncio
import traceback
import uvicorn
nest_asyncio.apply()
from dotenv import load_dotenv
load_dotenv()
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# File paths
TOKEN_FILE = "token_list.json"
DATA_DIR = "vector_data"

# In-memory storage for indexes and query engines
indexes = {}
query_engines = {}

# Ensure necessary directories and files exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(TOKEN_FILE):
    with open(TOKEN_FILE, "w") as f:
        json.dump({}, f)

try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass


def load_token_list():
    """Load token list from file."""
    with open(TOKEN_FILE, "r") as f:
        return json.load(f)


def save_token_list(token_list):
    """Save token list to file."""
    with open(TOKEN_FILE, "w") as f:
        json.dump(token_list, f, indent=4)

class DeleteIndexRequest(BaseModel):
    token: str

class MetadataFilterRequest(BaseModel):
    key: str
    value: Union[str, int, float, List[str], List[int], List[float], None] = None
    operator: str = Field(default="==")

    @validator('value')
    def validate_value(cls, v):
        # Additional custom validation can be added here
        if v is None:
            return v
        
        # Ensure lists are homogeneous
        if isinstance(v, list):
            if not v:
                return v
            
            # Determine the type of the first element
            base_type = type(v[0])
            
            # Ensure all elements are of the same type
            if not all(isinstance(item, base_type) for item in v):
                raise ValueError("All list elements must be of the same type")
        
        return v

# Add new nested filter structures
class NestedFilterGroup(BaseModel):
    condition: str = Field(default="AND", pattern="^(AND|OR)$")
    filters: List[Union[MetadataFilterRequest, Dict]] = [] # Can accept both types

class IndexRequest(BaseModel):
    index_name: str


class VectorAddRequest(BaseModel):
    token: str
    content: List[str]
    metadata: dict

class DeleteVectorRequest(BaseModel):
    token: str
    metadata: dict

class UpdateVectorsRequest(BaseModel):
    token: str
    content: List[str]  # List of strings representing the new vector content
    metadata: dict  # Metadata associated with the new content


class QueryRequest(BaseModel):
    token: str
    query: str
    query_engine_id: Optional[str] = None
    prompt: Optional[str] = None
    metadata_filters: Optional[List[MetadataFilterRequest]] = None
    filter_groups: Optional[List[NestedFilterGroup]] = None
    top_level_condition: Optional[str] = Field(default="AND", pattern="^(AND|OR)$")


class QueryEngineRequest(BaseModel):
    token: str
    query_engine_name: str
    metadata_filters: Optional[List[MetadataFilterRequest]] = None  # For backwards compatibility
    filter_groups: Optional[List[NestedFilterGroup]] = None  # For nested filters
    top_level_condition: Optional[str] = Field(default="AND", pattern="^(AND|OR)$")

class QueryEngineReloadRequest(BaseModel):
    token: str
    query_engine_id: str

def get_query_keywords(text: str) -> List[str]:
    
    """
    Enhanced keyword extraction with better Nepali support
    """
    # Check if the text contains Nepali characters
    is_nepali = any(ord(c) > 128 for c in text.lower())
    
    if is_nepali:
        # For Nepali text, split on whitespace and keep more words
        # Don't remove stopwords as Nepali stopwords might be significant
        words = text.lower().split()
        # Remove very short words (less than 2 chars) and duplicates
        keywords = list(set([word for word in words if len(word) > 2]))
        return keywords
    else:
        # Original English processing
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [word for word in tokens if word not in stop_words and len(word) > 2]
        return list(set(keywords))

def create_llamaindex_filters(
    metadata_filters: Optional[List[Union[MetadataFilterRequest, Dict]]] = None,
    filter_groups: Optional[List[Union[NestedFilterGroup, Dict]]] = None,
    top_level_condition: str = "AND"
) -> Optional[MetadataFilters]:
    """
    Helper function to create LlamaIndex MetadataFilters from either simple or nested filters.
    Handles both object and dictionary formats.
    """
    try:
        if metadata_filters:
            # Handle simple filters
            filters = []
            for f in metadata_filters:
                if isinstance(f, dict):
                    # Handle dictionary format
                    filter_dict = f
                else:
                    # Handle object format
                    filter_dict = f.dict()
                
                filters.append(
                    MetadataFilter(
                        key=filter_dict["key"],
                        value=filter_dict["value"],
                        operator=FilterOperator(filter_dict.get("operator", "=="))
                    )
                )
            return MetadataFilters(filters=filters)
        
        elif filter_groups:
            # Handle nested filters
            filter_groups_list = []
            
            for group in filter_groups:
                # Convert group to dict if it's an object
                group_dict = group if isinstance(group, dict) else group.dict()
                
                group_filters = []
                for f in group_dict.get("filters", []):
                    # Convert filter to dict if it's an object
                    filter_dict = f if isinstance(f, dict) else f.dict()
                    
                    group_filters.append(
                        MetadataFilter(
                            key=filter_dict["key"],
                            value=filter_dict["value"],
                            operator=FilterOperator(filter_dict.get("operator", "=="))
                        )
                    )
                
                group_condition = FilterCondition.AND if group_dict.get("condition", "AND").upper() == "AND" else FilterCondition.OR
                filter_groups_list.append(
                    MetadataFilters(
                        filters=group_filters,
                        condition=group_condition
                    )
                )
            
            top_condition = FilterCondition.AND if top_level_condition.upper() == "AND" else FilterCondition.OR
            return MetadataFilters(
                filters=filter_groups_list,
                condition=top_condition
            )
        
        return None

    except Exception as e:
        # Log the specific error and the input data for debugging
        print(f"Error in create_llamaindex_filters: {str(e)}")
        print(f"Input metadata_filters: {metadata_filters}")
        print(f"Input filter_groups: {filter_groups}")
        print(f"Input top_level_condition: {top_level_condition}")
        raise

@app.on_event("startup")
async def initialize_indexes_and_query_engines():
    """
    Load all indexes and query engines into memory during server startup.
    """
    token_list = load_token_list()
    for token, token_info in token_list.items():
        try:
            # Load the index with CustomVectorStore
            index_name = token_info["index_name"]
            index_path = os.path.join(DATA_DIR, index_name)
            
            # Initialize CustomVectorStore and load data
            vector_store = CustomVectorStore()
            vector_store.load(index_path)  # Load existing vectors if any
            
            # Create storage context with custom vector store
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=index_path
            )
            
            # Load index from storage
            index = load_index_from_storage(storage_context)
            indexes[token] = index
            
            print(f"Loaded index {index_name} with {len(vector_store._data)} vectors")

            # Recreate query engines
            if "query_engines" in token_info:
                query_engines[token] = {}
                for query_engine_id, engine_info in token_info["query_engines"].items():
                    try:
                        filters = create_llamaindex_filters(
                            metadata_filters=engine_info.get("metadata_filters"),
                            filter_groups=engine_info.get("filter_groups"),
                            top_level_condition=engine_info.get("top_level_condition", "AND")
                        )
                        
                        # Create query engine with filters
                        response_synthesizer = get_response_synthesizer(
                            structured_answer_filtering=True
                        )
                        
                        query_engine = index.as_query_engine(
                            filters=filters,
                            response_synthesizer=response_synthesizer,
                            response_mode="tree_summarize"
                        )
                        
                        query_engines[token][query_engine_id] = query_engine
                        print(f"Recreated query engine {query_engine_id} with filters: {filters}")

                    except Exception as e:
                        print(f"Failed to recreate query engine {query_engine_id}: {str(e)}")
                        print(traceback.format_exc())

        except Exception as e:
            print(f"Failed to load index {index_name}: {str(e)}")
            print(traceback.format_exc())


@app.post("/create_index/")
async def create_index(request: IndexRequest):
    token_list = load_token_list()

    if request.index_name in token_list:
        raise HTTPException(status_code=400, detail="Index already exists.")

    token = str(uuid.uuid4())
    index_path = os.path.join(DATA_DIR, request.index_name)
    
    if not os.path.exists(index_path):
        os.makedirs(index_path)

    # Initialize with CustomVectorStore
    vector_store = CustomVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex([], storage_context=storage_context, embed_model=OpenAIEmbedding())
    
    index.storage_context.persist(index_path)
    indexes[token] = index

    token_list[token] = {"token": token, "index_name": request.index_name}
    save_token_list(token_list)

    return {"message": f"Index {request.index_name} created successfully.", "token": token}


@app.get("/list_tokens/")
async def list_tokens():
    return load_token_list()

@app.get("/list_indexes/")
async def list_indexes():
    """
    List all indexes from memory.
    """
    return str(indexes)


@app.post("/list_query_engines/")
async def list_query_engines():
    """
    List all query engines for an index identified by the token.
    """
    return str(query_engines)

# @app.post("/create_query_engine/")
# async def create_query_engine(request: QueryEngineRequest):
#     if request.token not in indexes:
#         raise HTTPException(status_code=404, detail="Invalid token.")

#     try:
#         index = indexes[request.token]
        
#         print("\n=== Creating Query Engine ===")
#         print(f"Query Engine Name: {request.query_engine_name}")
        
#         # Create filters
#         filters = create_llamaindex_filters(
#             metadata_filters=request.metadata_filters,
#             filter_groups=request.filter_groups,
#             top_level_condition=request.top_level_condition
#         )
        
#         # if filters:
#         #     print("\nApplied Filters:")
#         #     print(f"Filter condition: {filters.condition}")
#         #     for f in filters.filters:
#         #         print(f"Filter: key={f.key}, value={f.value}, operator={f.operator}")

#         if filters:
#             print("\nApplied Filters:")
#             print(f"Filter condition: {filters.condition}")
#             # Correctly handle nested filters
#             for filter_item in filters.filters:
#                 if isinstance(filter_item, MetadataFilter):
#                     print(f"Filter: key={filter_item.key}, value={filter_item.value}, operator={filter_item.operator}")
#                 elif isinstance(filter_item, MetadataFilters):
#                     print(f"Filter Group: condition={filter_item.condition}")
#                     for sub_filter in filter_item.filters:
#                         print(f"  Sub-filter: key={sub_filter.key}, value={sub_filter.value}, operator={sub_filter.operator}")


#         # Generate ID
#         query_engine_id = generate_hash(str(request.dict()))
        
#         # Configure LLM with better parameters for query understanding
#         llm = OpenAI(
#             model="gpt-4o-mini",
#             temperature=0.0,
#             max_tokens=1500,
#             top_p=0.95,
#             presence_penalty=0.3,  # Increase diversity in responses
#             frequency_penalty=0.3   # Reduce repetition
#         )
        
#         # Configure retriever for better search
#         retriever = VectorIndexRetriever(
#             index=index,
#             similarity_top_k=5,  # Set similarity_top_k here
#             filters=filters
#         )
        
#         # Configure response synthesizer with better parameters
#         response_synthesizer = get_response_synthesizer(
#             response_mode="tree_summarize",
#             llm=llm,
#             structured_answer_filtering=True
#         )
        
#         # Create query engine with the configured components
#         query_engine = RetrieverQueryEngine(
#             retriever=retriever,
#             response_synthesizer=response_synthesizer,
#             node_postprocessors=[
#                 SimilarityPostprocessor(similarity_cutoff=0.6)
#             ]
#         )

#         # Store query engine
#         if request.token not in query_engines:
#             query_engines[request.token] = {}
#         query_engines[request.token][query_engine_id] = query_engine

#         # Save configuration
#         token_list = load_token_list()
#         if request.token in token_list:
#             if "query_engines" not in token_list[request.token]:
#                 token_list[request.token]["query_engines"] = {}
            
#             engine_config = {
#                 "name": request.query_engine_name
#             }
            
#             if request.metadata_filters:
#                 engine_config["metadata_filters"] = [
#                     {
#                         "key": f.key,
#                         "value": f.value,
#                         "operator": f.operator
#                     } for f in request.metadata_filters
#                 ]
            
#             if request.filter_groups:
#                 engine_config["filter_groups"] = [
#                     {
#                         "condition": group.condition,
#                         "filters": [
#                             {
#                                 "key": f.key,
#                                 "value": f.value,
#                                 "operator": f.operator
#                             } for f in group.filters
#                         ]
#                     } for group in request.filter_groups
#                 ]
#                 engine_config["top_level_condition"] = request.top_level_condition
            
#             token_list[request.token]["query_engines"][query_engine_id] = engine_config
#             save_token_list(token_list)
        
#         print(f"\nQuery Engine Created Successfully")
#         print(f"Engine ID: {query_engine_id}")
        
#         return {
#             "message": "Query engine created successfully.",
#             "query_engine_id": query_engine_id
#         }

#     except Exception as e:
#         print(f"Error creating query engine: {str(e)}")
#         print(traceback.format_exc())
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error creating query engine: {str(e)}"
#         )

@app.post("/create_query_engine/")
async def create_query_engine(request: QueryEngineRequest):
    if request.token not in indexes:
        raise HTTPException(status_code=404, detail="Invalid token.")

    try:
        index = indexes[request.token]
        
        print("\n=== Creating Query Engine ===")
        print(f"Query Engine Name: {request.query_engine_name}")
        
        # Create filters
        filters = create_llamaindex_filters(
            metadata_filters=request.metadata_filters,
            filter_groups=request.filter_groups,
            top_level_condition=request.top_level_condition
        )

        # Print filter information for debugging
        if filters:
            print("\nApplied Filters:")
            print(f"Filter condition: {filters.condition}")
            for filter_item in filters.filters:
                if isinstance(filter_item, MetadataFilter):
                    print(f"Filter: key={filter_item.key}, value={filter_item.value}, operator={filter_item.operator}")
                elif isinstance(filter_item, MetadataFilters):
                    print(f"Filter Group: condition={filter_item.condition}")
                    for sub_filter in filter_item.filters:
                        print(f"  Sub-filter: key={sub_filter.key}, value={sub_filter.value}, operator={sub_filter.operator}")

        # Generate ID
        query_engine_id = generate_hash(str(request.dict()))
        
        # Configure LLM
        llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2000,
            top_p=0.95,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )

        # Configure embedding model
        embed_model = OpenAIEmbedding()

        class HybridVectorRetriever(VectorIndexRetriever):
            """Custom retriever that combines semantic and keyword search."""
            
            def __init__(self, index, similarity_top_k=5, filters=None, alpha=0.5):
                """Initialize the hybrid retriever."""
                super().__init__(index=index, similarity_top_k=similarity_top_k, filters=filters)
                self.alpha = alpha
                self.index = index

            def _retrieve(self, query_bundle: QueryBundle):
                """Retrieve nodes using both semantic and keyword matching."""
                try:
                    # Get semantic search results using parent's retrieve method
                    semantic_results = super()._retrieve(query_bundle)
                    print(f"Number of semantic results: {len(semantic_results)}")
                    
                    # Extract keywords from the query
                    keywords = get_query_keywords(query_bundle.query_str)
                    print(f"Extracted keywords from query: {keywords}")
                    
                    # Score nodes based on more flexible keyword matches
                    keyword_scores = {}
                    for node in semantic_results:
                        # Convert both text and keywords to lower case for comparison
                        node_text = node.node.text.lower()
                        # More flexible matching for each keyword
                        keyword_matches = 0
                        for keyword in keywords:
                            keyword_lower = keyword.lower()
                            # Check for exact match or partial match
                            if (keyword_lower in node_text or 
                                any(kw in keyword_lower or keyword_lower in kw 
                                    for kw in node_text.split())):
                                keyword_matches += 1
                        
                        # Calculate normalized score
                        keyword_scores[node.node.node_id] = keyword_matches / len(keywords) if keywords else 0
                        
                        # Debug print for this node
                        if keyword_matches > 0:
                            print(f"Node ID: {node.node.node_id}")
                            print(f"Keyword matches: {keyword_matches}")
                            print(f"Text preview: {node.node.text[:100]}...")
                    
                    # Combine semantic and keyword scores with more weight on keyword matches
                    combined_results = []
                    for node in semantic_results:
                        node_id = node.node.node_id
                        semantic_score = node.score if hasattr(node, 'score') else 0.0
                        keyword_score = keyword_scores.get(node_id, 0.0)
                        
                        # Adjust the weights to favor keyword matches more
                        combined_score = (1.0 * semantic_score + 0.0 * keyword_score)
                        
                        # Only include if there's some relevance
                        if combined_score > 0:
                            node.score = combined_score
                            combined_results.append(node)
                            print(f"Added node with score {combined_score}")
                    
                    # Sort by combined score
                    combined_results.sort(key=lambda x: x.score, reverse=True)
                    results = combined_results[:self.similarity_top_k]
                    
                    print(f"Final number of results: {len(results)}")
                    return results if results else semantic_results[:self.similarity_top_k]
                
                except Exception as e:
                    print(f"Error in hybrid retrieval: {str(e)}")
                    print(traceback.format_exc())
                    return semantic_results  # Fall back to semantic results on error

        class CustomKeywordPostprocessor(BaseNodePostprocessor):
            """Custom postprocessor that dynamically handles keyword matching."""
            
            def __init__(self):
                """Initialize the postprocessor."""
                super().__init__()

            def _postprocess_nodes(
                self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
            ) -> List[NodeWithScore]:
                """Post process nodes by checking keyword presence."""
                if not query_bundle or not query_bundle.query_str:
                    return nodes
                    
                query_str = query_bundle.query_str
                is_nepali = any(ord(c) > 128 for c in query_str)

                try:
                    # Extract keywords from the query
                    keywords = get_query_keywords(query_str)
                    print(f"Processing keywords for {'Nepali' if is_nepali else 'English'} query: {keywords}")

                    # If no meaningful keywords found, return original nodes
                    if not keywords:
                        return nodes

                    # Process all nodes and adjust scores
                    processed_nodes = []
                    for node in nodes:
                        node_text = node.node.text.lower()
                        matching_keywords = 0
                        keyword_matches = []
                        
                        # Check for keyword matches
                        for keyword in keywords:
                            if is_nepali:

                                #for Nepali, check if keyword appears anywhere in the text.
                                if keyword in node_text:
                                    matching_keywords += 1
                                    keyword_matches.append(keyword) 

                                    #Checks for partial word matches in Nepali
                                    #This helps with different forms of the same word
                                elif any(word.startswith(keyword) or word.endswith(keyword) for word in node_text.split()):
                                    matching_keywords += 0.7  #Partial matches gets 0.7 weight
                                    keyword_matches.append(f"{keyword}(partial)")
                            
                            else:
                                # Check for partial matches
                                if keyword in node_text:
                                    matching_keywords += 1
                                    keyword_matches.append(keyword)
                                
                        
                        # Always keep the node, but adjust its score
                        if matching_keywords > 0:
                            base_score = matching_keywords / len(keywords)

                            #for nepali queries , boost scores of nodes with good keyword matches
                            if is_nepali and base_score > 0.5:
                                base_score *= 1.2 #20% boost for good nepali matches

                            if hasattr(node, 'score'):
                                #Combine with original score, weighting keywords are more heavily for Nepali
                                keyword_weight = 0.7 if is_nepali else 0.5 
                                node.score = (node.score * (1- keyword_weight) + base_score * keyword_weight)
                            else:
                                node.score = base_score
                            print(f"Node Scored {node.score} with matches: {keyword_matches}")
                            processed_nodes.append(node)
                        else:
                            # Keep nodes with no keyword matches but with lower scores
                            if hasattr(node, 'score'):
                                node.score = node.score * 0.3  # Reduce score for non-matching nodes
                            else:
                                node.score = 0.1  # Assign a low base score
                            processed_nodes.append(node)
                    
                    # Sort by score and return
                    processed_nodes.sort(key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)

                    #For Nepali queries, we can be more lenient with the minimum score
                    min_score = 0.15 if is_nepali else 0.3
                    return [n for n in processed_nodes if getattr(n, 'score', 0) > min_score]

                except Exception as e:
                    print(f"Error in postprocessing: {str(e)}")
                    print(traceback.format_exc())
                    return nodes

        # Create hybrid retriever
        retriever = HybridVectorRetriever(
            index=index,
            similarity_top_k=5,  # Increased for better recall
            filters=filters,
            alpha=0.2  
        )
        
        # Configure node postprocessors
        node_postprocessors = [
            CustomKeywordPostprocessor(),
            # NaturalLanguagePostProcessor(),  # Add the new post-processor
            SimilarityPostprocessor(similarity_cutoff=0.2)
        ]

        # Configure settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 20
        
        # Configure response synthesizer
        # Configure response synthesizer with natural language templates
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            structured_answer_filtering=True,
            text_qa_template= PromptTemplate( template ="""
            Context information is below.
            ---------------------
            {context_str}
            ---------------------
            Generate a detailed educational response for this query: {query_str}
            
            Response Guidelines:
            1. Language Choice:
                - If the query is in Nepali, respond in clear, formal Nepali
                - If in English, use simple academic English
                - Use appropriate educational terms in both languages
            
            2. Content Structure:
                - Start with a clear introduction of the topic
                - Present information in a logical, structured manner
                - Use examples or explanations when needed
                - End with a concise summary
            
            3. Educational Elements:
                - Include relevant course details, prerequisites, or requirements
                - Mention any important deadlines or schedules
                - Explain technical terms or concepts when present
                - Reference specific learning outcomes where applicable
            
            4. Cultural Context:
                - Include relevant context for Nepal's education system
                - Use local examples when appropriate
                - Consider both traditional and modern learning approaches
            
            5. Student Support:
                - Address common student concerns
                - Include information about available resources
                - Mention relevant study materials or tools
                - Provide clear next steps if applicable
            
            Maintain an encouraging, supportive tone throughout the response while ensuring accuracy and completeness.
            Response: """),
            refine_template= PromptTemplate( template = """
                Original query: {query_str}
        
                Previous response: {existing_answer}
                
                New context information: {context_msg}
                
                Guidelines for refinement:
                1. Language Consistency:
                    - Maintain the same language (Nepali or English) as the original response
                    - Keep consistent terminology and tone
                
                2. Educational Focus:
                    - Integrate new academic information seamlessly
                    - Update course details or requirements if needed
                    - Maintain clear learning objectives
                
                3. Content Integration:
                    - Blend new information naturally into existing content
                    - Update examples or explanations if necessary
                    - Ensure logical flow between old and new information
                
                4. Student Support Updates:
                    - Update resource information if relevant
                    - Add any new important deadlines or requirements
                    - Include additional guidance if needed
                
                5. Quality Checks:
                    - Verify all dates and deadlines are current
                    - Ensure course codes and names are accurate
                    - Confirm all prerequisites are clearly stated
                    - Check that all technical terms are explained
                
                Create a refined response that maintains educational value while incorporating new information.
                If the new context isn't relevant to the educational query, maintain the original response.
            Updated response: """)

            
        )
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors
        )

        # Store query engine
        if request.token not in query_engines:
            query_engines[request.token] = {}
        query_engines[request.token][query_engine_id] = query_engine

        # Save configuration
        token_list = load_token_list()
        if request.token in token_list:
            if "query_engines" not in token_list[request.token]:
                token_list[request.token]["query_engines"] = {}
            
            engine_config = {
                "name": request.query_engine_name,
                "hybrid_search": {
                    "semantic_weight": 0.3,
                    "keyword_weight": 0.7,
                    "top_k": 10
                }
            }
            
            if request.metadata_filters:
                engine_config["metadata_filters"] = [
                    {
                        "key": f.key,
                        "value": f.value,
                        "operator": f.operator
                    } for f in request.metadata_filters
                ]
            
            if request.filter_groups:
                engine_config["filter_groups"] = [
                    {
                        "condition": group.condition,
                        "filters": [
                            {
                                "key": f.key,
                                "value": f.value,
                                "operator": f.operator
                            } for f in group.filters
                        ]
                    } for group in request.filter_groups
                ]
                engine_config["top_level_condition"] = request.top_level_condition
            
            token_list[request.token]["query_engines"][query_engine_id] = engine_config
            save_token_list(token_list)
        
        print(f"\nQuery Engine Created Successfully")
        print(f"Engine ID: {query_engine_id}")
        
        return {
            "message": "Enhanced query engine with hybrid search created successfully.",
            "query_engine_id": query_engine_id,
            "config": engine_config
        }

    except Exception as e:
        print(f"Error creating query engine: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error creating query engine: {str(e)}"
        )

@app.delete("/query_engine/")
async def delete_query_engine(request: QueryRequest):
    """
    Delete a query engine from an index identified by the token.
    """
    try:
        # Validate the token
        if request.token not in query_engines:
            raise HTTPException(status_code=404, detail="No query engines found for the given token.")

        # Validate the query engine ID
        if request.query_engine_id not in query_engines[request.token]:
            raise HTTPException(status_code=404, detail="Query engine ID not found.")

        # Delete the query engine from memory
        del query_engines[request.token][request.query_engine_id]

        # Update the token list
        token_list = load_token_list()
        for index_name, index_info in token_list.items():
            if index_info["token"] == request.token:
                del index_info["query_engines"][request.query_engine_id]
                break

        save_token_list(token_list)

        return {"message": "Query engine deleted successfully."}
    except Exception as e:
        return {"error": str(e)}


@app.delete("/index/")
async def delete_index(request: DeleteIndexRequest):
    """
    Delete an index and all related components (query engines, vectors, storage) identified by the token.
    """
    token_list = load_token_list()

    # Validate the token
    if request.token not in token_list:
        raise HTTPException(status_code=404, detail="Invalid token.")

    try:
        print("\n=== Starting Index Deletion Process ===")
        deletion_report = {"components_deleted": {}}
        
        # Get index information before deletion
        index_name = token_list[request.token]["index_name"]
        index_path = os.path.join(DATA_DIR, index_name)
        print(f"Processing deletion for index: {index_name}")

        # 1. Delete query engines
        engines_deleted = 0
        if request.token in query_engines:
            engines_deleted = len(query_engines[request.token])
            del query_engines[request.token]
            deletion_report["components_deleted"]["query_engines"] = {
                "count": engines_deleted,
                "status": "successfully_deleted"
            }
        print(f"Deleted {engines_deleted} query engines")

        # 2. Delete vectors and nodes from memory
        vectors_deleted = 0
        if request.token in indexes:
            index = indexes[request.token]
            vectors_deleted = len(index.vector_store._data)
            
            # Clear vector store data
            index.vector_store._data.clear()
            index.vector_store._metadata.clear()
            
            # Clear docstore
            index.docstore.docs.clear()
            
            # Remove from memory
            del indexes[request.token]
            
            deletion_report["components_deleted"]["vectors"] = {
                "count": vectors_deleted,
                "status": "successfully_deleted"
            }
        print(f"Deleted {vectors_deleted} vectors")

        # 3. Delete physical storage
        storage_deleted = False
        if os.path.exists(index_path):
            try:
                # Delete the entire directory and its contents
                import shutil
                shutil.rmtree(index_path)
                storage_deleted = True
                
                # Also check and delete any auxiliary files
                vector_file = os.path.join(index_path, "vectors.json")
                metadata_file = os.path.join(index_path, "metadata.json")
                
                deletion_report["components_deleted"]["storage"] = {
                    "path": index_path,
                    "status": "successfully_deleted",
                    "auxiliary_files": [
                        "vectors.json",
                        "metadata.json"
                    ]
                }
            except Exception as storage_error:
                print(f"Error deleting storage: {str(storage_error)}")
                deletion_report["components_deleted"]["storage"] = {
                    "status": "error",
                    "error": str(storage_error)
                }
        print(f"Storage deletion status: {storage_deleted}")

        # 4. Remove token from token list
        del token_list[request.token]
        save_token_list(token_list)
        deletion_report["components_deleted"]["token"] = {
            "token": request.token,
            "status": "successfully_deleted"
        }
        print("Token removed from token list")

        # Final cleanup check
        deletion_report["verification"] = {
            "token_exists": request.token in token_list,
            "index_exists": request.token in indexes,
            "query_engines_exist": request.token in query_engines,
            "storage_exists": os.path.exists(index_path)
        }
        
        deletion_report["status"] = "success"
        deletion_report["summary"] = {
            "index_name": index_name,
            "total_vectors_deleted": vectors_deleted,
            "total_engines_deleted": engines_deleted,
            "storage_deleted": storage_deleted
        }
        
        print("=== Index Deletion Complete ===\n")
        
        return deletion_report

    except Exception as e:
        print(f"Error during index deletion: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail={
                "error": f"Error during index deletion: {str(e)}",
                "partial_deletion": deletion_report if 'deletion_report' in locals() else None
            }
        )


class DeleteNodesRequest(BaseModel):
    token: str
    metadata_filters: List[MetadataFilterRequest] = None                                # For backward compatibility
    filter_groups: Optional[List[NestedFilterGroup]] = None                             # For nested filters
    top_level_condition: Optional[str] = Field(default="AND", pattern="^(AND|OR)$")


@app.post("/delete_nodes/")
async def delete_nodes(request: DeleteNodesRequest):
    if request.token not in indexes:
        raise HTTPException(status_code=404, detail="Invalid token.")

    try:
        index: VectorStoreIndex = indexes[request.token]
        vector_store: CustomVectorStore = index.vector_store
        
        # Create filters based on request type
        if request.filter_groups:
            # Handle nested filters
            filters = create_llamaindex_filters(
                filter_groups=request.filter_groups,
                top_level_condition=request.top_level_condition
            )
        else:
            # Handle simple metadata filters for backward compatibility
            filters = create_llamaindex_filters(
                metadata_filters=request.metadata_filters
            )

        nodes_to_delete = []
        deleted_nodes_info = []  # Store detailed info about deleted nodes
        
        # Use CustomVectorStore's evaluate_filters method
        for node_id, metadata in vector_store._metadata.items():
            if vector_store.evaluate_filters(metadata, filters):
                nodes_to_delete.append(node_id)
                # Store node information before deletion
                node_info = {
                    "node_id": node_id,
                    "metadata": metadata,
                    "content": vector_store._data[node_id]["text"] if "text" in vector_store._data[node_id] else None
                }
                deleted_nodes_info.append(node_info)

        # Delete matched nodes
        if nodes_to_delete:
            # Delete from vector store
            vector_store.delete(nodes_to_delete)
            # Delete from docstore
            for node_id in nodes_to_delete:
                if node_id in index.docstore.docs:
                    del index.docstore.docs[node_id]

        # Persist changes
        token_list = load_token_list()
        index_path = os.path.join(DATA_DIR, token_list[request.token]["index_name"])
        index.storage_context.persist(index_path)

        return {
            "message": f"Successfully deleted {len(nodes_to_delete)} nodes.",
            "deleted_count": len(nodes_to_delete),
            "deleted_node_ids": nodes_to_delete,
            "filter_applied": {
                "filter_groups": request.filter_groups,
                "metadata_filters": request.metadata_filters,
                "top_level_condition": request.top_level_condition
            },
            "deleted_nodes_details": [
                {
                    "node_id": info["node_id"],
                    "metadata": info["metadata"],
                    "content_preview": info["content"][:200] + "..." if info["content"] else None  # First 200 chars of content
                }
                for info in deleted_nodes_info
            ],
            "storage_status": {
                "remaining_vectors": len(vector_store._data),
                "remaining_metadata": len(vector_store._metadata),
                "remaining_docs": len(index.docstore.docs)
            }
        }

    except Exception as e:
        print(f"Error in delete_nodes: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/add_vector/")
async def add_vector(request: VectorAddRequest):
    """
    Add vectors to the index and ensure they're properly stored in vector store
    """
    token_list = load_token_list()
    
    # Validate token
    if not request.token in token_list:
        raise HTTPException(status_code=404, detail="Invalid token.")
        
    index_name = token_list[request.token]["index_name"]
    index_path = os.path.join(DATA_DIR, index_name)
    index = indexes[request.token]
    
    try:
        print("\nStarting vector addition process...")
        print(f"Current vector store size: {len(index.vector_store._data)}")
        print(f"Current docstore size: {len(index.docstore.docs)}")
        
        # Generate embeddings
        embed_model = OpenAIEmbedding()
        nodes = []
        
        for content in request.content:
            print(f"\nProcessing content: {content[:100]}...")
            
            # Generate embedding
            embedding = embed_model.get_text_embedding(content)
            print(f"Generated embedding of size: {len(embedding)}")
            
            # Create node
            node_id = str(uuid.uuid4())
            node = TextNode(
                text=content,
                metadata=request.metadata,
                embedding=embedding,
                id_=node_id,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=node_id),
                }
            )
            nodes.append(node)
            print(f"Created node with ID: {node_id}")
            print(f"Node metadata: {request.metadata}")
            
            # Add directly to vector store's internal storage
            index.vector_store._data[node_id] = {
                "vector": embedding,
                "text": content,
            }
            index.vector_store._metadata[node_id] = request.metadata
            
            # Add to docstore
            index.docstore.add_documents([node])
        
        print(f"\nAfter addition:")
        print(f"Vector store size: {len(index.vector_store._data)}")
        print(f"Vector store metadata size: {len(index.vector_store._metadata)}")
        print(f"Docstore size: {len(index.docstore.docs)}")
        
        # Persist changes
        print(f"\nPersisting to {index_path}...")
        if not os.path.exists(index_path):
            os.makedirs(index_path)
            
        # Persist vector store data
        vector_file = os.path.join(index_path, "vectors.json")
        metadata_file = os.path.join(index_path, "metadata.json")
        
        # Save vectors
        with open(vector_file, 'w') as f:
            json.dump(index.vector_store._data, f)
        print(f"Saved vectors to {vector_file}")
            
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(index.vector_store._metadata, f)
        print(f"Saved metadata to {metadata_file}")
        
        # Persist entire storage context
        index.storage_context.persist(persist_dir=index_path)
        print("Storage context persisted")
        
        return {
            "message": "Vectors added successfully",
            "added_count": len(nodes),
            "vector_store_size": len(index.vector_store._data),
            "docstore_size": len(index.docstore.docs),
            "node_ids": [node.id_ for node in nodes]
        }
        
    except Exception as e:
        print(f"Error in add_vector: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload_query_engine/")
async def reload_query_engine(request: QueryEngineReloadRequest):
    """
    Reload a query engine for an index identified by the token.
    """
    if request.token not in query_engines:
        raise HTTPException(status_code=404, detail="No query engines found for token")

    token_list = load_token_list()
    engine_info = token_list[request.token]["query_engines"].get(request.query_engine_id)
    if not engine_info:
        raise HTTPException(status_code=404, detail="Query engine not found")

    # Create new request with saved configuration
    reload_request = QueryEngineRequest(
        token=request.token,
        query_engine_name=engine_info["name"],
        metadata_filters=[MetadataFilterRequest(**f) for f in engine_info.get("metadata_filters", [])] if "metadata_filters" in engine_info else None,
        filter_groups=[NestedFilterGroup(**g) for g in engine_info.get("filter_groups", [])] if "filter_groups" in engine_info else None,
        top_level_condition=engine_info.get("top_level_condition", "AND")
    )

    # Create new query engine
    response = await create_query_engine(reload_request)
    return {"message": "Query engine reloaded successfully"}



def generate_hash(filters: Union[List[Dict], str, List[NestedFilterGroup]]) -> str:
    """
    Generate a unique hash for metadata filters or filter groups.
    
    Args:
        filters: Can be:
            - List[Dict] for simple metadata filters
            - str for raw string representation
            - List[NestedFilterGroup] for nested filter groups
    
    Returns:
        str: A unique hash string
    """
   
    
    try:
        if isinstance(filters, str):
            # Handle raw string input
            hash_input = filters
        elif isinstance(filters, list):
            if filters and isinstance(filters[0], NestedFilterGroup):
                # Handle nested filter groups
                hash_input = json.dumps([{
                    "condition": group.condition,
                    "filters": [{
                        "key": f.key if isinstance(f, MetadataFilterRequest) else f["key"],
                        "value": f.value if isinstance(f, MetadataFilterRequest) else f["value"],
                        "operator": f.operator if isinstance(f, MetadataFilterRequest) else f.get("operator", "==")
                    } for f in group.filters]
                } for group in filters], sort_keys=True)
            else:
                # Handle simple metadata filters
                hash_input = json.dumps([{
                    "key": f.get("key"),
                    "value": f.get("value"),
                    "operator": f.get("operator", "==")
                } for f in filters], sort_keys=True)
        else:
            # Handle unexpected input
            hash_input = str(filters)
        
        # Generate hash using SHA-256
        return hashlib.sha256(hash_input.encode()).hexdigest()
    except Exception as e:
        # If there's any error in processing, fall back to string representation
        return hashlib.sha256(str(filters).encode()).hexdigest()


# @app.post("/query_index/")
# async def query_index(request: QueryRequest):
#     """
#     Query with enhanced debugging and logging
#     """
#     if request.token not in query_engines:
#         raise HTTPException(status_code=404, detail="No query engines found for token")

#     try:
#         # Debug logging
#         print("\n=== Query Debug Information ===")
#         print(f"Query: {request.query}")
#         print(f"Token: {request.token}")
        
#         # Check vector store data
#         index = indexes[request.token]
#         vector_store = index.vector_store
#         print(f"\nVector Store Stats:")
#         print(f"Total vectors: {len(vector_store._data)}")
#         print(f"Total metadata entries: {len(vector_store._metadata)}")
        
#         # Print some sample metadata if available
#         if vector_store._metadata:
#             print("\nSample Metadata Entry:")
#             sample_id = next(iter(vector_store._metadata))
#             print(f"ID: {sample_id}")
#             print(f"Metadata: {vector_store._metadata[sample_id]}")

#         # Create or get query engine
#         if request.query_engine_id:
#             query_engine_id = request.query_engine_id
#         else:
#             temp_request = QueryEngineRequest(
#                 token=request.token,
#                 query_engine_name="temporary_engine",
#                 metadata_filters=request.metadata_filters,
#                 filter_groups=request.filter_groups,
#                 top_level_condition=request.top_level_condition
#             )
#             resp = await create_query_engine(temp_request)
#             query_engine_id = resp["query_engine_id"]

#         print(f"\nQuery Engine ID: {query_engine_id}")
        
#         query_engine = query_engines[request.token].get(query_engine_id)
#         if not query_engine:
#             raise HTTPException(status_code=404, detail="Query engine not found")

#         # Print filter information
#         if request.metadata_filters:
#             print("\nApplied Metadata Filters:")
#             for f in request.metadata_filters:
#                 print(f"Key: {f.key}, Value: {f.value}, Operator: {f.operator}")

#         if request.filter_groups:
#             print("\nApplied Filter Groups:")
#             for group in request.filter_groups:
#                 print(f"Group Condition: {group.condition}")
#                 for f in group.filters:
#                     print(f"  Key: {f.key}, Value: {f.value}, Operator: {f.operator}")

#         # Execute query with custom prompt if provided
#         if request.prompt:
#             response = query_engine.query(request.query, request.prompt)
#         else:
#             response = query_engine.query(request.query)

#         print("\nResponse Information:")
#         print(f"Response type: {type(response)}")
#         print(f"Response content: {response.response}")
#         print(f"Number of source nodes: {len(response.source_nodes)}")

#         # Process source nodes
#         source_nodes = []
#         for source_node in response.source_nodes:
#             try:
#                 if hasattr(source_node, 'node'):
#                     node_data = {
#                         "score": float(source_node.score) if hasattr(source_node, 'score') else None,
#                         "text": source_node.node.text,
#                         "metadata": source_node.node.metadata,
#                         "id": source_node.node.id_
#                     }
#                 else:
#                     node_data = {
#                         "score": None,
#                         "text": source_node.text,
#                         "metadata": source_node.metadata,
#                         "id": source_node.id_
#                     }
#                 source_nodes.append(node_data)
#             except Exception as node_error:
#                 print(f"Error processing node: {str(node_error)}")
#                 continue

#         result = {
#             "query": request.query,
#             "response": response.response if response.response else "No relevant information found.",
#             "source_nodes": source_nodes,
#             "metadata": {
#                 "engine_id": query_engine_id,
#                 "total_nodes": len(source_nodes),
#                 "response_metadata": getattr(response, 'metadata', {}),
#                 "vector_store_size": len(vector_store._data)
#             }
#         }

#         print("\n=== End Debug Information ===")
#         return result

#     except Exception as e:
#         print(f"\nError during query: {str(e)}")
#         print(traceback.format_exc())
#         raise HTTPException(
#             status_code=500,
#             detail=f"Query error: {str(e)}"
#         )

@app.post("/query_index/")
async def query_index(request: QueryRequest):
    """
    Enhanced query endpoint with comprehensive debugging and direct error handling
    """
    if request.token not in query_engines:
        raise HTTPException(status_code=404, detail="No query engines found for token")

    try:
        # Enhanced Debug logging
        print("\n=== Query Execution Debug ===")
        print(f"Query: {request.query}")
        print(f"Token: {request.token}")
        
        # Vector store validation
        index = indexes[request.token]
        vector_store = index.vector_store
        if not vector_store._data:
            return {
                "query": request.query,
                "response": "The vector store is empty. Please add some data first.",
                "source_nodes": [],
                "metadata": {"status": "empty_vector_store"}
            }
            
        print(f"\nVector Store Status:")
        print(f"Total vectors: {len(vector_store._data)}")
        print(f"Total metadata entries: {len(vector_store._metadata)}")
        print(f"Sample vector dimensions: {len(next(iter(vector_store._data.values()))['vector'])}")

        # Query Engine Setup
        if request.query_engine_id:
            query_engine_id = request.query_engine_id
            print(f"Using existing query engine: {query_engine_id}")
        else:
            # Create temporary engine with debugging
            print("\nCreating temporary query engine")
            temp_request = QueryEngineRequest(
                token=request.token,
                query_engine_name="temporary_engine",
                metadata_filters=request.metadata_filters,
                filter_groups=request.filter_groups,
                top_level_condition=request.top_level_condition
            )
            resp = await create_query_engine(temp_request)
            query_engine_id = resp["query_engine_id"]
            print(f"Created temporary engine: {query_engine_id}")

        query_engine = query_engines[request.token].get(query_engine_id)
        if not query_engine:
            raise HTTPException(status_code=404, detail="Query engine not found")

        # Debug Filter Information
        if request.metadata_filters:
            print("\nActive Metadata Filters:")
            for f in request.metadata_filters:
                print(f"Filter - Key: {f.key}, Value: {f.value}, Operator: {f.operator}")
                # Check if any nodes match this filter
                matching_nodes = [
                    node_id for node_id, metadata in vector_store._metadata.items()
                    if vector_store.evaluate_filter(metadata, MetadataFilter(
                        key=f.key, value=f.value, operator=FilterOperator(f.operator)
                    ))
                ]
                print(f"Nodes matching this filter: {len(matching_nodes)}")

        # Execute Query with Response Validation
        try:
            if request.prompt:
                print("\nExecuting query with custom prompt")
                custom_prompt_template = PromptTemplate(template= request.prompt)
                response = query_engine.query(request.query, text_qa_template=custom_prompt_template)
            else:
                print("\nExecuting query with default prompt")
                response = query_engine.query(request.query)
            
            print(f"Raw response type: {type(response)}")
            print(f"Response content length: {len(response.response) if response.response else 0}")
            
            # Handle Empty Response
            if not response or not response.response:
                print("\nEmpty response detected")
                
                # Get raw nodes for debugging
                retriever = query_engine.retriever
                raw_nodes = retriever.retrieve(request.query)
                print(f"Retrieved {len(raw_nodes)} raw nodes")
                
                if not raw_nodes:
                    return {
                        "query": request.query,
                        "response": "No relevant information found. Please try refining your query or check your metadata filters.",
                        "source_nodes": [],
                        "metadata": {
                            "status": "no_matching_nodes",
                            "vector_store_size": len(vector_store._data),
                            "filter_info": {
                                "metadata_filters": [
                                    {"key": f.key, "value": f.value, "operator": f.operator}
                                    for f in (request.metadata_filters or [])
                                ],
                                "filter_groups": request.filter_groups
                            } if request.metadata_filters or request.filter_groups else None
                        }
                    }
                
                return {
                    "query": request.query,
                    "response": "Query returned no response. Retrieved nodes are available in source_nodes.",
                    "source_nodes": [
                        {
                            "score": getattr(node, 'score', None),
                            "text": node.node.text,
                            "metadata": node.node.metadata,
                            "id": node.node.id_
                        } for node in raw_nodes
                    ],
                    "metadata": {
                        "status": "empty_response_with_nodes",
                        "engine_id": query_engine_id,
                        "vector_store_size": len(vector_store._data)
                    }
                }

            # Process Valid Response
            source_nodes = []
            for source_node in response.source_nodes:
                try:
                    node_data = {
                        "score": float(source_node.score) if hasattr(source_node, 'score') else None,
                        "text": source_node.node.text if hasattr(source_node, 'node') else source_node.text,
                        "metadata": source_node.node.metadata if hasattr(source_node, 'node') else source_node.metadata,
                        "id": source_node.node.id_ if hasattr(source_node, 'node') else source_node.id_
                    }
                    source_nodes.append(node_data)
                except Exception as node_error:
                    print(f"Error processing node: {str(node_error)}")
                    continue

            return {
                "query": request.query,
                "response": response.response,
                "source_nodes": source_nodes,
                "metadata": {
                    "status": "success",
                    "engine_id": query_engine_id,
                    "total_nodes": len(source_nodes),
                    "response_metadata": getattr(response, 'metadata', {}),
                    "vector_store_size": len(vector_store._data)
                }
            }

        except Exception as query_error:
            print(f"Query execution error: {str(query_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Query execution failed: {str(query_error)}"
            )

    except Exception as e:
        print(f"General error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Query processing error: {str(e)}"
        )
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5601)
