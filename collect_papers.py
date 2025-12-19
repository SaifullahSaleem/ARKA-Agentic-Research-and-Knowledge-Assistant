"""
Script to collect research papers from Google Scholar using SERP API
and save metadata to CSV/JSON format.
"""

import os
import json
import csv
import time
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
import requests
from tqdm import tqdm

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
PAPERS_DIR = "papers"
METADATA_FILE = "paper_metadata.json"
METADATA_CSV = "paper_metadata.csv"

# Domain-specific search queries
DOMAIN = "Machine Learning in Healthcare"
SEARCH_QUERIES = [
    "machine learning healthcare diagnosis",
    "deep learning medical imaging",
    "AI clinical decision support",
    "neural networks electronic health records",
    "predictive analytics healthcare",
    "ML drug discovery",
    "AI radiology pathology",
    "healthcare NLP clinical notes",
    "reinforcement learning treatment optimization",
    "federated learning medical data",
    "transformer models healthcare",
    "computer vision medical diagnosis",
    "time series analysis patient monitoring",
    "ML precision medicine",
    "AI healthcare ethics",
]


def create_directories():
    """Create necessary directories."""
    os.makedirs(PAPERS_DIR, exist_ok=True)
    os.makedirs("pdfs", exist_ok=True)


def search_google_scholar(query: str, num_results: int = 10) -> List[Dict]:
    """
    Search Google Scholar for papers.
    
    Args:
        query: Search query string
        num_results: Number of results to fetch
        
    Returns:
        List of paper metadata dictionaries
    """
    papers = []
    
    try:
        # Use SerpAPI REST API directly
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": min(num_results, 20),  # Max 20 per request
        }
        
        # Make API request
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        results = response.json()
        
        if "organic_results" in results:
            for item in results["organic_results"]:
                paper = {
                    "title": item.get("title", ""),
                    "authors": ", ".join([a.get("name", "") for a in item.get("publication_info", {}).get("authors", [])]),
                    "year": item.get("publication_info", {}).get("year", ""),
                    "abstract": item.get("snippet", ""),
                    "url": item.get("link", ""),
                    "cited_by": item.get("inline_links", {}).get("cited_by", {}).get("total", 0),
                    "doi": "",
                    "source": "google_scholar",
                    "query_used": query,
                    "collected_at": datetime.now().isoformat(),
                }
                
                # Try to extract DOI from URL or other fields
                if "doi" in item.get("link", "").lower():
                    paper["doi"] = item.get("link", "")
                
                papers.append(paper)
        
        time.sleep(1)  # Rate limiting
        
    except Exception as e:
        print(f"Error searching for '{query}': {str(e)}")
    
    return papers


def download_pdf(url: str, filename: str) -> bool:
    """
    Attempt to download PDF from URL.
    
    Args:
        url: URL of the paper
        filename: Local filename to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10, stream=True)
        
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
    
    return False


def collect_papers(min_papers: int = 35) -> List[Dict]:
    """
    Collect papers from Google Scholar until we have at least min_papers.
    
    Args:
        min_papers: Minimum number of papers to collect
        
    Returns:
        List of unique paper metadata
    """
    all_papers = []
    seen_titles = set()
    
    print(f"Collecting papers for domain: {DOMAIN}")
    print(f"Target: At least {min_papers} papers\n")
    
    for query in tqdm(SEARCH_QUERIES, desc="Searching queries"):
        papers = search_google_scholar(query, num_results=10)
        
        for paper in papers:
            title_lower = paper["title"].lower().strip()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                all_papers.append(paper)
                
                if len(all_papers) >= min_papers:
                    break
        
        if len(all_papers) >= min_papers:
            break
        
        time.sleep(2)  # Rate limiting between queries
    
    # If we still need more papers, do additional searches
    if len(all_papers) < min_papers:
        additional_queries = [
            "ML healthcare 2023",
            "AI medicine 2024",
            "deep learning clinical",
        ]
        
        for query in additional_queries:
            if len(all_papers) >= min_papers:
                break
            papers = search_google_scholar(query, num_results=15)
            for paper in papers:
                title_lower = paper["title"].lower().strip()
                if title_lower and title_lower not in seen_titles:
                    seen_titles.add(title_lower)
                    all_papers.append(paper)
                    if len(all_papers) >= min_papers:
                        break
            time.sleep(2)
    
    return all_papers[:min_papers] if len(all_papers) > min_papers else all_papers


def save_metadata(papers: List[Dict]):
    """Save paper metadata to JSON and CSV files."""
    # Save as JSON
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    # Save as CSV
    if papers:
        fieldnames = papers[0].keys()
        with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(papers)
    
    print(f"\nSaved {len(papers)} papers to {METADATA_FILE} and {METADATA_CSV}")


def main():
    """Main function to collect papers."""
    create_directories()
    
    print("=" * 60)
    print("Research Paper Collection Script")
    print("=" * 60)
    
    papers = collect_papers(min_papers=35)
    
    if papers:
        save_metadata(papers)
        
        # Statistics
        print("\n" + "=" * 60)
        print("Collection Statistics:")
        print("=" * 60)
        print(f"Total papers collected: {len(papers)}")
        print(f"Papers from Google Scholar: {sum(1 for p in papers if p.get('source') == 'google_scholar')}")
        print(f"Papers with abstracts: {sum(1 for p in papers if p.get('abstract'))}")
        print(f"Average citations: {sum(p.get('cited_by', 0) for p in papers) / len(papers):.1f}")
        
        # Year distribution
        years = [p.get('year', 'Unknown') for p in papers]
        year_counts = {}
        for year in years:
            year_counts[year] = year_counts.get(year, 0) + 1
        print(f"\nYear distribution: {dict(sorted(year_counts.items(), reverse=True)[:5])}")
    else:
        print("No papers collected. Please check your API key and internet connection.")


if __name__ == "__main__":
    main()

