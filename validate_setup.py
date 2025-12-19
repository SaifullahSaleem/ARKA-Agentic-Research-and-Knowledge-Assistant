"""
Validation script to check if all components are working correctly
"""

import os
import sys
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"⚠️  {description}: {dirpath} - NOT FOUND (will be created)")
        return False

def validate_environment():
    """Validate environment setup."""
    print("=" * 60)
    print("Environment Validation")
    print("=" * 60)
    
    issues = []
    
    # Check .env file
    if not os.path.exists(".env"):
        print("❌ .env file not found")
        issues.append("Create .env file with API keys")
    else:
        print("✅ .env file exists")
        
        # Check API keys
        from dotenv import load_dotenv
        load_dotenv()
        
        if not os.getenv("GROQ_API_KEY"):
            print("❌ GROQ_API_KEY not set in .env")
            issues.append("Set GROQ_API_KEY in .env")
        else:
            print("✅ GROQ_API_KEY is set")
        
        if not os.getenv("SERPAPI_KEY"):
            print("❌ SERPAPI_KEY not set in .env")
            issues.append("Set SERPAPI_KEY in .env")
        else:
            print("✅ SERPAPI_KEY is set")
    
    return len(issues) == 0, issues

def validate_data_files():
    """Validate data files."""
    print("\n" + "=" * 60)
    print("Data Files Validation")
    print("=" * 60)
    
    required_files = [
        ("paper_metadata.json", "Paper metadata"),
        ("document_chunks.pkl", "Document chunks"),
        ("embeddings.npy", "Embeddings"),
        ("faiss_index.bin", "FAISS index"),
        ("chunks_metadata.json", "Chunks metadata"),
    ]
    
    all_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist

def validate_code_files():
    """Validate code files."""
    print("\n" + "=" * 60)
    print("Code Files Validation")
    print("=" * 60)
    
    required_files = [
        ("collect_papers.py", "Paper collection script"),
        ("preprocessing_notebook.py", "Preprocessing script"),
        ("agentic_interface.py", "Agentic interface"),
        ("evaluation_metrics.py", "Evaluation metrics"),
        ("experimentation_suite.py", "Experimentation suite"),
        ("app.py", "Streamlit app"),
        ("config.yaml", "Configuration file"),
        ("requirements.txt", "Requirements file"),
    ]
    
    all_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist

def validate_dependencies():
    """Validate Python dependencies."""
    print("\n" + "=" * 60)
    print("Dependencies Validation")
    print("=" * 60)
    
    required_packages = [
        "langchain",
        "langchain_groq",
        "sentence_transformers",
        "faiss",
        "streamlit",
        "pandas",
        "numpy",
        "prometheus_client",
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def validate_config():
    """Validate configuration file."""
    print("\n" + "=" * 60)
    print("Configuration Validation")
    print("=" * 60)
    
    try:
        import yaml
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        required_keys = ["domain", "vector_store", "retrieval", "llm", "agent", "evaluation", "monitoring"]
        
        for key in required_keys:
            if key in config:
                print(f"✅ {key}")
            else:
                print(f"❌ {key} - MISSING")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error reading config.yaml: {e}")
        return False

def main():
    """Main validation function."""
    print("\n" + "=" * 60)
    print("System Validation")
    print("=" * 60)
    print()
    
    results = {
        "environment": validate_environment(),
        "code_files": validate_code_files(),
        "dependencies": validate_dependencies(),
        "config": validate_config(),
        "data_files": validate_data_files(),
    }
    
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    all_passed = True
    for component, (passed, issues) in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{component.upper()}: {status}")
        if issues:
            for issue in issues:
                print(f"  - {issue}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All validations passed! System is ready.")
    else:
        print("⚠️  Some validations failed. Please fix the issues above.")
        print("\nNext steps:")
        if not results["environment"][0]:
            print("1. Create .env file with API keys")
        if not results["dependencies"][0]:
            print("2. Install missing dependencies: pip install -r requirements.txt")
        if not results["data_files"][0]:
            print("3. Run: python collect_papers.py")
            print("4. Run: python preprocessing_notebook.py")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

