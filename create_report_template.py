"""
Template generator for the final report
Creates a structured report template with sections for all requirements.
"""

report_template = """# Agentic AI Interface from Research Papers
## Final Report

**Domain:** Machine Learning in Healthcare  
**Date:** {date}  
**Author:** [Your Name]

---

## 1. Introduction

### 1.1 Domain Selection
[Explain why you chose Machine Learning in Healthcare as the domain. Discuss its relevance and the availability of research papers.]

### 1.2 Objectives
[State the main objectives of the project: building an agentic AI interface, evaluation, and monitoring.]

---

## 2. Literature Corpus

### 2.1 Collection Methodology
[Describe how papers were collected using Google Scholar and SERP API. Mention the search queries used and collection process.]

### 2.2 Corpus Statistics
- **Total Papers:** 35+
- **Source:** Google Scholar (via SERP API)
- **Year Range:** [Fill in]
- **Average Citations:** [Fill in]
- **Papers with PDFs:** [Fill in]

### 2.3 Metadata
[Describe the metadata structure: title, authors, year, abstract, DOI/URL, source indicator.]

---

## 3. System Architecture

### 3.1 Data Preprocessing Pipeline
[Describe PDF extraction, chunking strategy, cleaning, and embedding generation.]

**Chunking Strategy:**
- Chunk size: 1000 tokens
- Overlap: 200 tokens
- Method: RecursiveCharacterTextSplitter

**Embeddings:**
- Model: sentence-transformers/all-MiniLM-L6-v2
- Dimension: 384

### 3.2 Retrieval System
[Describe FAISS vector store, similarity search, and retrieval logic.]

### 3.3 Agentic Interface
[Describe LangChain agent, tool use, and LLM orchestration.]

**LLM Configuration:**
- Provider: Groq
- Model: llama-3.1-70b-versatile
- Temperature: 0.7
- Max tokens: 2048

### 3.4 Web Interface
[Describe Streamlit UI and user interaction flow.]

---

## 4. Evaluation Metrics

### 4.1 Context Precision (CP)
**Definition:** CP = (# of retrieved chunks that contain facts directly relevant to the answer) / k

**Methodology:**
- Manual labeling of relevant chunks for 30+ queries
- Gold standard creation process
- Inter-annotator agreement (if applicable)

**Results:**
[Present results with tables/plots]

### 4.2 Answer Relevance (AR)
**Definition:** Human rating from 1 (not relevant) to 5 (perfectly relevant)

**Methodology:**
- Labeling guidelines provided to annotators
- Examples of each rating level
- Automatic metrics (Jaccard similarity) as baseline

**Results:**
[Present results with tables/plots]

### 4.3 Faithfulness / Hallucination Rate
**Definition:** Fraction of factual claims in the model output that cannot be corroborated by retrieved chunks.

**Methodology:**
- Sentence-level claim extraction
- Word overlap with context (30% threshold)
- NLI model for entailment checking (optional)

**Results:**
[Present results with tables/plots]

### 4.4 Chunk Retrieval Statistics
**Metrics:**
- Distribution of k
- Average overlap ratio
- Average token coverage
- Number of unique papers contributing to top-k

**Results:**
[Present results with tables/plots]

### 4.5 Efficiency Metrics
**Metrics:**
- Latency (ms)
- Tokens consumed
- Cost estimates
- Tokens per second

**Results:**
[Present results with tables/plots]

---

## 5. Experiments

### 5.1 Hyperparameter Grid
[Describe the parameter grid tested: k, chunk_size, temperature, model.]

### 5.2 Experimental Setup
[Describe the experimental procedure, number of queries, evaluation protocol.]

### 5.3 Results
[Present comprehensive results with tables and visualizations.]

**Key Findings:**
- [Finding 1]
- [Finding 2]
- [Finding 3]

### 5.4 Analysis
[Discuss trade-offs, optimal configurations, and insights.]

---

## 6. Monitoring and Observability

### 6.1 Prometheus Metrics
[Describe the metrics exposed: query rate, latency, tokens, chunks, active queries.]

### 6.2 Grafana Dashboards
[Describe the dashboard panels and visualizations.]

### 6.3 Performance Insights
[Discuss insights from monitoring: bottlenecks, optimization opportunities.]

---

## 7. Discussion

### 7.1 Strengths
[What worked well?]

### 7.2 Limitations
[What are the limitations of the current system?]

### 7.3 Challenges
[What challenges were encountered during development?]

### 7.4 Lessons Learned
[Key takeaways from the project.]

---

## 8. Recommendations

### 8.1 System Improvements
[Suggestions for improving retrieval, generation, or evaluation.]

### 8.2 Future Work
[Ideas for extending the system: multi-modal support, advanced agents, etc.]

---

## 9. Conclusion

[Summarize the project, achievements, and contributions.]

---

## 10. References

[List all research papers used in the corpus, tools, and frameworks.]

---

## Appendix

### A. Labeling Guidelines
[Detailed guidelines for human annotators.]

### B. Code Repository
[Link to code repository if applicable.]

### C. Additional Visualizations
[Extra plots and figures.]

"""

def create_report():
    """Create the report template file."""
    from datetime import datetime
    
    content = report_template.format(date=datetime.now().strftime("%Y-%m-%d"))
    
    with open("FINAL_REPORT.md", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("Report template created: FINAL_REPORT.md")
    print("Please fill in the sections with your results and analysis.")

if __name__ == "__main__":
    create_report()

