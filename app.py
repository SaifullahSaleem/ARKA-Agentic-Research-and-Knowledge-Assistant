"""
Streamlit Web UI for the Agentic AI Interface
"""

import streamlit as st
import json
from datetime import datetime
from agentic_interface import get_interface
from evaluation_metrics import EvaluationMetrics
import logging

# Configure page
st.set_page_config(
    page_title="Agentic AI Research Interface",
    page_icon="🔬",
    layout="wide",
)

# Global style theme (light gradient, accent colors, card styling)
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background: radial-gradient(circle at 20% 20%, #f6f9ff 0, #f9fbff 35%, #fefefe 80%);
        color: #1f2937;
    }
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #1b3764 !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.25);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8, #1e40af);
    }
    /* Cards */
    .stExpander, .stMetric, .stTextInput, .stSelectbox, .stTextArea {
        border-radius: 10px;
    }
    /* Text area */
    textarea {
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
    }
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
    }
    /* Links */
    a {
        color: #2563eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize
if "interface" not in st.session_state:
    st.session_state.interface = get_interface()
    st.session_state.evaluator = EvaluationMetrics()
    st.session_state.query_history = []

# Title
st.title("🔬 Agentic AI Research Paper Interface")
st.markdown("**Domain:** Machine Learning in Healthcare")
st.markdown("Query research papers using natural language. The agent retrieves relevant context and generates comprehensive answers.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    k = st.slider("Number of chunks to retrieve (k)", 3, 10, 5)
    
    st.header("Query History")
    if st.button("Clear History"):
        st.session_state.query_history = []
    
    for i, item in enumerate(reversed(st.session_state.query_history[-10:])):
        with st.expander(f"Query {len(st.session_state.query_history) - i}: {item['query'][:50]}..."):
            st.write(f"**Answer:** {item['answer'][:200]}...")
            st.write(f"**Time:** {item['timestamp']}")

# Main interface
query = st.text_area(
    "Enter your query:",
    height=100,
    placeholder="e.g., What are the main applications of deep learning in medical imaging?",
)

col1, col2 = st.columns([1, 4])

with col1:
    submit_button = st.button("Submit Query", type="primary")

if submit_button and query:
    with st.spinner("Processing your query..."):
        try:
            # Process query
            result = st.session_state.interface.query(query)
            
            # Display answer
            st.subheader("Answer")
            st.write(result["answer"])
            
            # Display metadata
            with st.expander("Query Metadata"):
                metadata = result["metadata"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Latency", f"{metadata['latency_seconds']:.2f}s")
                with col2:
                    st.metric("Chunks Retrieved", metadata["num_chunks"])
                with col3:
                    st.metric("Unique Papers", metadata["num_unique_papers"])
                
                st.json(metadata)
            
            # Display retrieved chunks
            with st.expander("Retrieved Context Chunks"):
                for i, chunk in enumerate(result["context_chunks"], 1):
                    st.markdown(f"**Chunk {i}** (Similarity: {chunk['similarity']:.3f})")
                    st.markdown(f"*Paper:* {chunk['paper']}")
                    st.markdown(f"*Text:* {chunk['text']}")
                    st.divider()
            
            # Evaluation metrics
            with st.expander("Evaluation Metrics"):
                # Use full retrieved_chunks for evaluation (if available), otherwise use context_chunks
                chunks_for_eval = result.get("retrieved_chunks", result["context_chunks"])
                evaluation = st.session_state.evaluator.evaluate_query(
                    query=query,
                    answer=result["answer"],
                    retrieved_chunks=chunks_for_eval,
                    latency_seconds=metadata["latency_seconds"],
                    tokens_used=metadata["estimated_tokens"],
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Faithfulness", f"{evaluation['faithfulness']['faithfulness_score']:.2%}")
                with col2:
                    st.metric("Hallucination Rate", f"{evaluation['faithfulness']['hallucination_rate']:.2%}")
                with col3:
                    st.metric("Supported Claims", f"{evaluation['faithfulness']['supported_claims']}/{evaluation['faithfulness']['total_claims']}")
                
                st.json(evaluation)
            
            # Save to history
            st.session_state.query_history.append({
                "query": query,
                "answer": result["answer"],
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata,
            })
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("**System Status:** ✅ Operational")
st.markdown(f"**Total Queries Processed:** {len(st.session_state.query_history)}")

