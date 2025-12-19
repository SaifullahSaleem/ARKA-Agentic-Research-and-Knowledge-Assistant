# Usage Guide

## Quick Start

1. **Setup Environment**
   ```bash
   # Windows
   setup.bat
   
   # Linux/Mac
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Collect Papers**
   ```bash
   python collect_papers.py
   ```
   This will create `paper_metadata.json` with at least 35 papers.

3. **Preprocess Data**
   ```bash
   python preprocessing_notebook.py
   ```
   This creates embeddings and FAISS index.

4. **Start Web Interface**
   ```bash
   streamlit run app.py
   ```

5. **Run Experiments** (Optional)
   ```bash
   python run_experiments.py
   ```

## Detailed Workflow

### Step 1: Paper Collection

The `collect_papers.py` script:
- Searches Google Scholar using SERP API
- Collects metadata (title, authors, year, abstract, URL)
- Saves to JSON and CSV formats
- Attempts to download PDFs (if available)

**Output:**
- `paper_metadata.json`
- `paper_metadata.csv`
- `pdfs/` directory (if PDFs are downloaded)

### Step 2: Data Preprocessing

The `preprocessing_notebook.py` script:
- Extracts text from PDFs (or uses abstracts)
- Chunks documents using RecursiveCharacterTextSplitter
- Generates embeddings using SentenceTransformers
- Creates FAISS index for fast similarity search
- Saves all processed data

**Output:**
- `document_chunks.pkl` - Text chunks
- `embeddings.npy` - Embedding vectors
- `faiss_index.bin` - FAISS index
- `chunks_metadata.json` - Chunk metadata with paper references

### Step 3: Query Interface

The Streamlit app (`app.py`) provides:
- Natural language query interface
- Real-time answer generation
- Retrieved context display
- Evaluation metrics
- Query history

**Features:**
- Adjustable k (number of chunks)
- View retrieved chunks with similarity scores
- See evaluation metrics (faithfulness, etc.)
- Track query history

### Step 4: Evaluation

**Manual Evaluation:**
1. Create test queries in `test_queries.json`
2. Run queries and collect results
3. Manually label relevant chunks for Context Precision
4. Rate answer relevance (1-5 scale)
5. Check faithfulness by verifying claims

**Automatic Evaluation:**
```bash
python run_experiments.py
```

This runs systematic experiments with different hyperparameters.

### Step 5: Monitoring

**Start Monitoring:**
```bash
python monitoring_setup.py
docker-compose up -d
```

**Access Dashboards:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

**Metrics Available:**
- Query rate
- Latency (p50, p95)
- Tokens consumed
- Chunks retrieved
- Active queries

## Configuration

Edit `config.yaml` to adjust:

```yaml
vector_store:
  chunk_size: 1000      # Size of text chunks
  chunk_overlap: 200    # Overlap between chunks
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

retrieval:
  k: 5                  # Number of chunks to retrieve
  search_type: "similarity"

llm:
  model: "llama-3.1-70b-versatile"
  temperature: 0.7
  max_tokens: 2048
```

## Troubleshooting

### API Key Issues
- Ensure `.env` file exists with correct keys
- Check API quotas and limits

### PDF Extraction Fails
- System automatically falls back to abstracts
- Some PDFs may be password-protected or have poor OCR

### Memory Issues
- Reduce chunk size or k
- Process papers in smaller batches

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.9+)

## Evaluation Guidelines

### Context Precision
1. For each query, identify which retrieved chunks are relevant
2. Label chunks as 1 (relevant) or 0 (not relevant)
3. Calculate: CP = (# relevant chunks) / k
4. Sample at least 30 queries for statistical significance

### Answer Relevance
- **1**: Not relevant - answer doesn't address the query
- **2**: Slightly relevant - touches on topic but misses key points
- **3**: Moderately relevant - addresses some aspects
- **4**: Very relevant - addresses most aspects well
- **5**: Perfectly relevant - comprehensive and accurate

### Faithfulness
1. Extract factual claims from the answer
2. Check if each claim is supported by retrieved chunks
3. Calculate: Faithfulness = (# supported claims) / (# total claims)

## Best Practices

1. **Query Formulation:**
   - Be specific and clear
   - Use domain-specific terminology
   - Ask focused questions

2. **Evaluation:**
   - Use consistent labeling guidelines
   - Have multiple annotators for reliability
   - Document edge cases

3. **Experimentation:**
   - Test systematically
   - Keep detailed logs
   - Compare against baselines

4. **Monitoring:**
   - Set up alerts for anomalies
   - Track trends over time
   - Use dashboards for debugging

