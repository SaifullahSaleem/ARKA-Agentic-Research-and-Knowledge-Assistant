"""
Evaluation Metrics Implementation
Context Precision, Answer Relevance, Faithfulness, Chunk Stats, Efficiency
"""

import json
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Compute evaluation metrics for the agentic AI system."""
    
    def __init__(self):
        # Load NLI model for faithfulness checking
        try:
            self.nli_model = pipeline(
                "text-classification",
                model="microsoft/deberta-v3-base",
                task="text-classification",
            )
        except Exception as e:
            logger.warning(f"Could not load NLI model: {e}. Using fallback method.")
            self.nli_model = None
    
    def context_precision(
        self,
        retrieved_chunks: List[Dict],
        relevant_chunk_indices: List[int],
        k: int
    ) -> float:
        """
        Compute Context Precision (CP).
        
        CP = (# of retrieved chunks that are relevant) / k
        
        Args:
            retrieved_chunks: List of retrieved chunks with metadata
            relevant_chunk_indices: Indices of chunks that are actually relevant (0-indexed)
            k: Number of chunks retrieved
            
        Returns:
            Context precision score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
        
        relevant_count = sum(1 for i in range(min(k, len(retrieved_chunks))) if i in relevant_chunk_indices)
        return relevant_count / k
    
    def answer_relevance(
        self,
        query: str,
        answer: str,
        ground_truth: Optional[str] = None
    ) -> Dict:
        """
        Compute Answer Relevance (AR).
        
        Returns human rating scale (1-5) and automatic metrics if ground truth available.
        
        Args:
            query: Original query
            answer: Generated answer
            ground_truth: Optional ground truth answer for automatic evaluation
            
        Returns:
            Dictionary with relevance score and metrics
        """
        result = {
            "human_rating": None,  # Should be filled by human annotator
            "automatic_score": None,
        }
        
        # Automatic evaluation if ground truth available
        if ground_truth:
            # Simple word overlap as baseline
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            ground_truth_words = set(ground_truth.lower().split())
            
            # Jaccard similarity
            answer_overlap = len(answer_words & ground_truth_words) / len(answer_words | ground_truth_words) if (answer_words | ground_truth_words) else 0
            
            result["automatic_score"] = {
                "jaccard_similarity": answer_overlap,
                "word_overlap_ratio": len(answer_words & ground_truth_words) / len(ground_truth_words) if ground_truth_words else 0,
            }
        
        return result
    
    def faithfulness(
        self,
        answer: str,
        retrieved_chunks: List[Dict]
    ) -> Dict:
        """
        Compute Faithfulness / Hallucination Rate.
        
        Checks if factual claims in the answer can be corroborated by retrieved chunks.
        
        Args:
            answer: Generated answer
            retrieved_chunks: List of retrieved chunks used to generate answer
            
        Returns:
            Dictionary with faithfulness metrics
        """
        # Extract claims from answer (simple sentence splitting)
        answer_sentences = [s.strip() for s in answer.split('.') if s.strip() and len(s.strip()) > 20]
        
        # Combine all retrieved chunks (handle both full chunks and simplified context_chunks)
        chunk_texts = []
        for chunk in retrieved_chunks:
            if "chunk_text" in chunk:
                chunk_texts.append(chunk["chunk_text"])
            elif "text" in chunk:
                # For simplified chunks, we need to get the full text
                # If we only have truncated text, use it
                chunk_texts.append(chunk["text"].replace("...", ""))
            else:
                # Fallback: try to get any text field
                chunk_texts.append(str(chunk.get("text", chunk.get("chunk_text", ""))))
        context = " ".join(chunk_texts)
        
        # Check each claim against context
        supported_claims = 0
        total_claims = len(answer_sentences)
        
        if total_claims == 0:
            return {
                "faithfulness_score": 1.0,
                "hallucination_rate": 0.0,
                "supported_claims": 0,
                "total_claims": 0,
            }
        
        for claim in answer_sentences:
            # Simple keyword matching (can be enhanced with NLI)
            claim_words = set(claim.lower().split())
            context_words = set(context.lower().split())
            
            # If significant overlap, consider claim supported
            overlap = len(claim_words & context_words)
            if overlap >= len(claim_words) * 0.3:  # At least 30% word overlap
                supported_claims += 1
        
        faithfulness_score = supported_claims / total_claims if total_claims > 0 else 1.0
        hallucination_rate = 1.0 - faithfulness_score
        
        return {
            "faithfulness_score": faithfulness_score,
            "hallucination_rate": hallucination_rate,
            "supported_claims": supported_claims,
            "total_claims": total_claims,
        }
    
    def chunk_retrieval_stats(
        self,
        retrieved_chunks: List[Dict],
        all_chunks_metadata: List[Dict]
    ) -> Dict:
        """
        Compute chunk retrieval statistics.
        
        Args:
            retrieved_chunks: List of retrieved chunks
            all_chunks_metadata: All available chunks metadata
            
        Returns:
            Dictionary with statistics
        """
        if not retrieved_chunks:
            return {}
        
        # Unique papers
        unique_papers = set()
        for chunk in retrieved_chunks:
            if "metadata" in chunk:
                paper_title = chunk.get("metadata", {}).get("paper_title", "Unknown")
            else:
                paper_title = chunk.get("paper", "Unknown")
            unique_papers.add(paper_title)
        
        # Chunk lengths (handle both structures)
        chunk_lengths = []
        for chunk in retrieved_chunks:
            if "chunk_text" in chunk:
                chunk_lengths.append(len(chunk.get("chunk_text", "")))
            elif "text" in chunk:
                chunk_lengths.append(len(chunk.get("text", "")))
            else:
                chunk_lengths.append(0)
        
        # Similarity scores
        similarity_scores = [chunk.get("similarity_score", chunk.get("similarity", 0.0)) for chunk in retrieved_chunks]
        
        # Compute overlap (simple token-based)
        chunk_texts = []
        for chunk in retrieved_chunks:
            if "chunk_text" in chunk:
                chunk_texts.append(chunk.get("chunk_text", ""))
            elif "text" in chunk:
                chunk_texts.append(chunk.get("text", "").replace("...", ""))
            else:
                chunk_texts.append("")
        all_tokens = set()
        for text in chunk_texts:
            all_tokens.update(text.lower().split())
        
        total_unique_tokens = len(all_tokens)
        total_tokens = sum(len(text.split()) for text in chunk_texts)
        overlap_ratio = 1.0 - (total_unique_tokens / total_tokens) if total_tokens > 0 else 0.0
        
        return {
            "num_chunks": len(retrieved_chunks),
            "num_unique_papers": len(unique_papers),
            "avg_chunk_length": np.mean(chunk_lengths) if chunk_lengths else 0,
            "min_chunk_length": np.min(chunk_lengths) if chunk_lengths else 0,
            "max_chunk_length": np.max(chunk_lengths) if chunk_lengths else 0,
            "avg_similarity": np.mean(similarity_scores) if similarity_scores else 0,
            "min_similarity": np.min(similarity_scores) if similarity_scores else 0,
            "max_similarity": np.max(similarity_scores) if similarity_scores else 0,
            "token_overlap_ratio": overlap_ratio,
            "unique_papers": list(unique_papers),
        }
    
    def efficiency_metrics(
        self,
        latency_seconds: float,
        tokens_used: int,
        cost_per_1k_tokens: float = 0.01
    ) -> Dict:
        """
        Compute efficiency metrics.
        
        Args:
            latency_seconds: Query latency in seconds
            tokens_used: Number of tokens consumed
            cost_per_1k_tokens: Cost per 1000 tokens (default estimate)
            
        Returns:
            Dictionary with efficiency metrics
        """
        return {
            "latency_ms": latency_seconds * 1000,
            "latency_seconds": latency_seconds,
            "tokens_consumed": tokens_used,
            "estimated_cost": (tokens_used / 1000) * cost_per_1k_tokens,
            "tokens_per_second": tokens_used / latency_seconds if latency_seconds > 0 else 0,
        }
    
    def evaluate_query(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict],
        latency_seconds: float,
        tokens_used: int,
        relevant_chunk_indices: Optional[List[int]] = None,
        ground_truth: Optional[str] = None,
        human_rating: Optional[int] = None
    ) -> Dict:
        """
        Comprehensive evaluation for a single query.
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_chunks: Retrieved chunks
            latency_seconds: Query latency
            tokens_used: Tokens consumed
            relevant_chunk_indices: Indices of relevant chunks (for CP)
            ground_truth: Ground truth answer (for AR)
            human_rating: Human relevance rating 1-5 (for AR)
            
        Returns:
            Complete evaluation results
        """
        k = len(retrieved_chunks)
        
        # Context Precision
        cp = 0.0
        if relevant_chunk_indices is not None:
            cp = self.context_precision(retrieved_chunks, relevant_chunk_indices, k)
        
        # Answer Relevance
        ar = self.answer_relevance(query, answer, ground_truth)
        if human_rating is not None:
            ar["human_rating"] = human_rating
        
        # Faithfulness
        faithfulness = self.faithfulness(answer, retrieved_chunks)
        
        # Chunk Stats
        chunk_stats = self.chunk_retrieval_stats(retrieved_chunks, [])
        
        # Efficiency
        efficiency = self.efficiency_metrics(latency_seconds, tokens_used)
        
        return {
            "query": query,
            "context_precision": cp,
            "answer_relevance": ar,
            "faithfulness": faithfulness,
            "chunk_stats": chunk_stats,
            "efficiency": efficiency,
            "timestamp": datetime.now().isoformat(),
        }
    
    def evaluate_batch(
        self,
        evaluation_results: List[Dict]
    ) -> Dict:
        """
        Aggregate evaluation results across multiple queries.
        
        Args:
            evaluation_results: List of evaluation results from evaluate_query()
            
        Returns:
            Aggregated statistics
        """
        if not evaluation_results:
            return {}
        
        cps = [r["context_precision"] for r in evaluation_results if r.get("context_precision") is not None]
        faithfulness_scores = [r["faithfulness"]["faithfulness_score"] for r in evaluation_results]
        latencies = [r["efficiency"]["latency_ms"] for r in evaluation_results]
        tokens = [r["efficiency"]["tokens_consumed"] for r in evaluation_results]
        
        human_ratings = [
            r["answer_relevance"]["human_rating"]
            for r in evaluation_results
            if r["answer_relevance"].get("human_rating") is not None
        ]
        
        return {
            "num_queries": len(evaluation_results),
            "context_precision": {
                "mean": np.mean(cps) if cps else 0,
                "std": np.std(cps) if cps else 0,
                "min": np.min(cps) if cps else 0,
                "max": np.max(cps) if cps else 0,
            },
            "faithfulness": {
                "mean": np.mean(faithfulness_scores) if faithfulness_scores else 0,
                "std": np.std(faithfulness_scores) if faithfulness_scores else 0,
            },
            "answer_relevance": {
                "mean_human_rating": np.mean(human_ratings) if human_ratings else None,
                "std_human_rating": np.std(human_ratings) if human_ratings else None,
            },
            "efficiency": {
                "mean_latency_ms": np.mean(latencies) if latencies else 0,
                "mean_tokens": np.mean(tokens) if tokens else 0,
                "total_tokens": sum(tokens) if tokens else 0,
            },
        }

