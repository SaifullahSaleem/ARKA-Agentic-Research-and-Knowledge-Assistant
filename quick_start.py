"""
Quick start script to test the system end-to-end
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def check_setup():
    """Check if system is properly set up."""
    print("=" * 60)
    print("Checking System Setup")
    print("=" * 60)
    
    issues = []
    
    # Check API keys
    if not os.getenv("GROQ_API_KEY"):
        issues.append("❌ GROQ_API_KEY not found in .env")
    else:
        print("✅ GROQ_API_KEY found")
    
    if not os.getenv("SERPAPI_KEY"):
        issues.append("❌ SERPAPI_KEY not found in .env")
    else:
        print("✅ SERPAPI_KEY found")
    
    # Check data files
    if not os.path.exists("paper_metadata.json"):
        issues.append("⚠️  paper_metadata.json not found - run collect_papers.py first")
    else:
        print("✅ paper_metadata.json found")
    
    if not os.path.exists("faiss_index.bin"):
        issues.append("⚠️  faiss_index.bin not found - run preprocessing_notebook.py first")
    else:
        print("✅ faiss_index.bin found")
    
    if not os.path.exists("document_chunks.pkl"):
        issues.append("⚠️  document_chunks.pkl not found - run preprocessing_notebook.py first")
    else:
        print("✅ document_chunks.pkl found")
    
    print("\n" + "=" * 60)
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\nPlease fix these issues before running the system.")
        return False
    else:
        print("✅ All checks passed! System is ready.")
        return True

def test_query():
    """Test a sample query."""
    print("\n" + "=" * 60)
    print("Testing Query Interface")
    print("=" * 60)
    
    try:
        from agentic_interface import get_interface
        
        interface = get_interface()
        test_query = "What are the main applications of deep learning in medical imaging?"
        
        print(f"\nQuery: {test_query}")
        print("Processing...")
        
        result = interface.query(test_query)
        
        print("\n✅ Query successful!")
        print(f"\nAnswer (first 200 chars): {result['answer'][:200]}...")
        print(f"Latency: {result['metadata']['latency_seconds']:.2f}s")
        print(f"Chunks retrieved: {result['metadata']['num_chunks']}")
        print(f"Unique papers: {result['metadata']['num_unique_papers']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Query test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    if check_setup():
        print("\n" + "=" * 60)
        response = input("Run test query? (y/n): ")
        if response.lower() == 'y':
            test_query()
    else:
        print("\nPlease complete the setup first:")
        print("1. python collect_papers.py")
        print("2. python preprocessing_notebook.py")
        print("3. streamlit run app.py")

if __name__ == "__main__":
    main()

