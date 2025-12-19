"""
Helper script to install requirements with better error handling for Python 3.13
"""

import subprocess
import sys

def install_package(package):
    """Install a single package with error handling."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install requirements with better error handling."""
    print("=" * 60)
    print("Installing Requirements for Python 3.13")
    print("=" * 60)
    print()
    
    # Core packages first
    core_packages = [
        "numpy>=2.0.0",
        "pandas>=2.2.3",
    ]
    
    print("Step 1: Installing core packages...")
    for package in core_packages:
        print(f"  Installing {package}...")
        install_package(package)
    
    # LLM packages
    llm_packages = [
        "langchain>=0.3.0",
        "langchain-community>=0.3.0",
        "langchain-groq>=0.2.0",
        "langchain-openai>=0.2.0",
        "langchain-text-splitters>=0.3.0",
        "langchain-core>=0.3.0",
        "groq>=0.4.1",
    ]
    
    print("\nStep 2: Installing LLM packages...")
    for package in llm_packages:
        print(f"  Installing {package}...")
        install_package(package)
    
    # Vector stores
    vector_packages = [
        "faiss-cpu>=1.13.1",
        "sentence-transformers>=2.7.0",
        "chromadb>=0.5.0",
    ]
    
    print("\nStep 3: Installing vector store packages...")
    for package in vector_packages:
        print(f"  Installing {package}...")
        install_package(package)
    
    # Remaining packages
    print("\nStep 4: Installing remaining packages from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError:
        print("⚠️  Some packages failed to install. Check errors above.")
        print("\nYou may need to install packages individually or update versions.")

if __name__ == "__main__":
    main()

