"""
Agentic AI Interface using LangChain
Implements retrieval, reasoning, and answer generation.
"""

import os
import json
import pickle
import numpy as np
import faiss
import yaml
from typing import List, Dict, Optional
from datetime import datetime
import uuid

from dotenv import load_dotenv
try:
    from langchain_groq import ChatGroq
except ImportError:
    try:
        # Fallback for older versions
        from langchain.llms import Groq as ChatGroq
    except ImportError:
        raise ImportError("Please install langchain-groq: pip install langchain-groq")

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
except ImportError:
    # Fallback for older versions
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging
from pythonjsonlogger import jsonlogger

load_dotenv()

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Prometheus metrics
query_counter = Counter('agent_queries_total', 'Total number of queries')
query_latency = Histogram('agent_query_latency_seconds', 'Query latency in seconds')
tokens_used = Counter('agent_tokens_total', 'Total tokens used')
chunks_retrieved = Histogram('agent_chunks_retrieved', 'Number of chunks retrieved per query')
active_queries = Gauge('agent_active_queries', 'Currently active queries')


class VectorStoreRetriever:
    """FAISS-based vector store retriever."""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(config["vector_store"]["embedding_model"])
        self.index = None
        self.chunks = None
        self.chunks_metadata = None
        self._load_data()
    
    def _load_data(self):
        """Load FAISS index and chunks."""
        try:
            # Load index
            self.index = faiss.read_index("faiss_index.bin")
            
            # Load chunks
            with open("document_chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
            
            # Load metadata
            with open("chunks_metadata.json", "r", encoding="utf-8") as f:
                self.chunks_metadata = json.load(f)
            
            logger.info("Vector store loaded successfully", extra={
                "chunks_count": len(self.chunks),
                "index_size": self.index.ntotal
            })
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks.
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with metadata
        """
        if k is None:
            k = config["retrieval"]["k"]
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                result = {
                    "chunk_text": self.chunks[idx],
                    "similarity_score": float(dist),
                    "metadata": self.chunks_metadata[idx] if idx < len(self.chunks_metadata) else {},
                }
                results.append(result)
        
        return results


class AgenticAIInterface:
    """Main agentic AI interface."""
    
    def __init__(self):
        self.retriever = VectorStoreRetriever()
        
        # Initialize LLM
        try:
            self.llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name=config["llm"]["model"],
                temperature=config["llm"]["temperature"],
                max_tokens=config["llm"]["max_tokens"],
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
        
        # Note: Using direct LLM calls with retrieval for simplicity
        # Full agent with tools can be added if needed
        
        # Start Prometheus metrics server
        try:
            start_http_server(config["monitoring"]["prometheus_port"])
            logger.info("Prometheus metrics server started")
        except Exception as e:
            logger.warning(f"Could not start Prometheus server: {e}")
    
    def _retrieve_context(self, query: str) -> str:
        """Retrieve context for the agent tool."""
        results = self.retriever.retrieve(query, k=config["retrieval"]["k"])
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}]: {result['chunk_text']}\n"
                f"Paper: {result['metadata'].get('paper_title', 'Unknown')}\n"
                f"Authors: {result['metadata'].get('paper_authors', 'Unknown')}\n"
            )
        
        return "\n".join(context_parts)
    
    def query(self, user_query: str, correlation_id: str = None) -> Dict:
        """
        Process a user query and return answer.
        
        Args:
            user_query: User's natural language query
            correlation_id: Optional correlation ID for logging
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        
        active_queries.inc()
        query_counter.inc()
        
        start_time = datetime.now()
        
        try:
            logger.info("Processing query", extra={
                "correlation_id": correlation_id,
                "query": user_query[:100]  # Truncate for logging
            })
            
            # Retrieve relevant context
            retrieved_chunks = self.retriever.retrieve(user_query, k=config["retrieval"]["k"])
            chunks_retrieved.observe(len(retrieved_chunks))
            
            # Build context
            context = "\n\n".join([chunk["chunk_text"] for chunk in retrieved_chunks])
            
            # Create prompt with context
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are an expert AI assistant specialized in {domain}. 
Use the following research paper excerpts to answer the question. 
If the information is not in the provided context, say so clearly.

Context from research papers:
{context}

Question: {question}

Provide a comprehensive answer based on the context. Cite the papers when relevant.
Answer:""".format(
                    domain=config["domain"],
                    context=context,
                    question=user_query
                )
            )
            
            # Generate answer
            prompt = prompt_template.format(context=context, question=user_query)
            
            # Handle both ChatGroq and regular LLM interfaces
            try:
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    answer = response.content
                elif isinstance(response, str):
                    answer = response
                else:
                    answer = str(response)
            except AttributeError:
                # Fallback for LLM interface without invoke
                answer = self.llm(prompt)
            
            # Estimate tokens (rough approximation)
            estimated_tokens = len(prompt.split()) + len(answer.split())
            tokens_used.inc(estimated_tokens)
            
            # Calculate latency
            latency = (datetime.now() - start_time).total_seconds()
            query_latency.observe(latency)
            
            # Get unique papers
            unique_papers = set()
            for chunk in retrieved_chunks:
                paper_title = chunk["metadata"].get("paper_title", "Unknown")
                unique_papers.add(paper_title)
            
            result = {
                "answer": answer,
                "context_chunks": [
                    {
                        "text": chunk["chunk_text"][:200] + "...",
                        "paper": chunk["metadata"].get("paper_title", "Unknown"),
                        "similarity": chunk["similarity_score"],
                    }
                    for chunk in retrieved_chunks
                ],
                "retrieved_chunks": retrieved_chunks,  # Full chunks for evaluation
                "metadata": {
                    "correlation_id": correlation_id,
                    "query": user_query,
                    "num_chunks": len(retrieved_chunks),
                    "num_unique_papers": len(unique_papers),
                    "latency_seconds": latency,
                    "estimated_tokens": estimated_tokens,
                    "timestamp": datetime.now().isoformat(),
                },
            }
            
            logger.info("Query completed", extra={
                "correlation_id": correlation_id,
                "latency": latency,
                "tokens": estimated_tokens,
                "chunks": len(retrieved_chunks)
            })
            
            return result
            
        except Exception as e:
            logger.error("Query failed", extra={
                "correlation_id": correlation_id,
                "error": str(e)
            })
            raise
        
        finally:
            active_queries.dec()


# Global instance
_interface = None


def get_interface() -> AgenticAIInterface:
    """Get or create the agentic interface instance."""
    global _interface
    if _interface is None:
        _interface = AgenticAIInterface()
    return _interface

