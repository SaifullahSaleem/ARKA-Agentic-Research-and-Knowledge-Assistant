"""
Data Preprocessing Notebook
Extracts text from PDFs, chunks documents, and generates embeddings.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import yaml

# PDF processing
from pypdf import PdfReader
import pdfplumber

# Embeddings and vector store
from sentence_transformers import SentenceTransformer
import faiss

# LangChain
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except ImportError:
    # Fallback for older versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document

load_dotenv()

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

PAPERS_DIR = "papers"
PDFS_DIR = "pdfs"
METADATA_FILE = "paper_metadata.json"
CHUNKS_FILE = "document_chunks.pkl"
EMBEDDINGS_FILE = "embeddings.npy"
INDEX_FILE = "faiss_index.bin"
CHUNKS_METADATA_FILE = "chunks_metadata.json"


class DocumentProcessor:
    """Process PDFs and create embeddings."""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(config["vector_store"]["embedding_model"])
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["vector_store"]["chunk_size"],
            chunk_overlap=config["vector_store"]["chunk_overlap"],
            length_function=len,
        )
        self.chunks = []
        self.embeddings = []
        self.chunks_metadata = []
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using multiple methods.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
                text = "\n\n".join(pages_text)
        except Exception as e:
            print(f"pdfplumber failed for {pdf_path}: {e}")
        
        # Fallback to pypdf
        if not text or len(text) < 100:
            try:
                reader = PdfReader(pdf_path)
                pages_text = []
                for page in reader.pages:
                    pages_text.append(page.extract_text())
                text = "\n\n".join(pages_text)
            except Exception as e:
                print(f"pypdf failed for {pdf_path}: {e}")
        
        return text
    
    def process_papers(self, metadata_file: str):
        """
        Process all papers from metadata file.
        
        Args:
            metadata_file: Path to paper metadata JSON file
        """
        with open(metadata_file, "r", encoding="utf-8") as f:
            papers = json.load(f)
        
        print(f"Processing {len(papers)} papers...")
        
        for idx, paper in enumerate(papers):
            print(f"\nProcessing paper {idx + 1}/{len(papers)}: {paper['title'][:50]}...")
            
            # Try to get PDF from URL or local file
            pdf_path = None
            text = ""
            
            # Check if PDF exists locally
            pdf_filename = f"paper_{idx + 1}.pdf"
            local_pdf = os.path.join(PDFS_DIR, pdf_filename)
            
            if os.path.exists(local_pdf):
                pdf_path = local_pdf
            elif paper.get("url"):
                # Try to download or use abstract
                pdf_path = paper["url"]
            
            # Extract text
            if pdf_path and os.path.exists(pdf_path):
                text = self.extract_text_from_pdf(pdf_path)
            
            # Fallback to abstract if PDF extraction fails
            if not text or len(text) < 100:
                text = paper.get("abstract", "")
                print(f"  Using abstract (PDF extraction failed or unavailable)")
            
            if not text or len(text) < 50:
                print(f"  Skipping: insufficient text")
                continue
            
            # Split into chunks
            documents = self.text_splitter.create_documents([text])
            
            # Add metadata to each chunk
            for doc in documents:
                chunk_metadata = {
                    "paper_title": paper["title"],
                    "paper_authors": paper.get("authors", ""),
                    "paper_year": paper.get("year", ""),
                    "paper_url": paper.get("url", ""),
                    "chunk_text": doc.page_content,
                    "chunk_length": len(doc.page_content),
                }
                
                self.chunks.append(doc.page_content)
                self.chunks_metadata.append(chunk_metadata)
        
        print(f"\nTotal chunks created: {len(self.chunks)}")
    
    def generate_embeddings(self):
        """Generate embeddings for all chunks."""
        print("\nGenerating embeddings...")
        
        if not self.chunks:
            raise ValueError("No chunks to embed. Run process_papers() first.")
        
        # Generate embeddings in batches
        batch_size = 32
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            
            if i == 0:
                self.embeddings = batch_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, batch_embeddings])
        
        print(f"Generated {len(self.embeddings)} embeddings of dimension {self.embeddings.shape[1]}")
    
    def create_faiss_index(self):
        """Create FAISS index for efficient similarity search."""
        print("\nCreating FAISS index...")
        
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings available. Run generate_embeddings() first.")
        
        dimension = self.embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Create index (Inner Product for cosine similarity after normalization)
        index = faiss.IndexFlatIP(dimension)
        index.add(self.embeddings.astype('float32'))
        
        # Save index
        faiss.write_index(index, INDEX_FILE)
        print(f"FAISS index saved to {INDEX_FILE}")
        
        return index
    
    def save_processed_data(self):
        """Save all processed data."""
        print("\nSaving processed data...")
        
        # Save chunks
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        
        # Save embeddings
        np.save(EMBEDDINGS_FILE, self.embeddings)
        
        # Save chunks metadata
        with open(CHUNKS_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, indent=2, ensure_ascii=False)
        
        print("All data saved successfully!")
    
    def get_statistics(self) -> Dict:
        """Get statistics about processed data."""
        if not self.chunks:
            return {}
        
        chunk_lengths = [len(chunk) for chunk in self.chunks]
        unique_papers = len(set(m["paper_title"] for m in self.chunks_metadata))
        
        return {
            "total_chunks": len(self.chunks),
            "unique_papers": unique_papers,
            "avg_chunk_length": np.mean(chunk_lengths),
            "min_chunk_length": np.min(chunk_lengths),
            "max_chunk_length": np.max(chunk_lengths),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
        }


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print("=" * 60)
    
    processor = DocumentProcessor()
    
    # Process papers
    if os.path.exists(METADATA_FILE):
        processor.process_papers(METADATA_FILE)
    else:
        print(f"Error: {METADATA_FILE} not found. Run collect_papers.py first.")
        return
    
    # Generate embeddings
    processor.generate_embeddings()
    
    # Create FAISS index
    processor.create_faiss_index()
    
    # Save everything
    processor.save_processed_data()
    
    # Print statistics
    stats = processor.get_statistics()
    print("\n" + "=" * 60)
    print("Processing Statistics:")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

