# Python 3.13 Compatibility Notes

## Overview

This project has been updated to work with **Python 3.13.3**. Some packages required version updates to ensure compatibility.

## Key Changes Made

### 1. Requirements Updates

- **numpy**: Updated to `>=2.0.0` (required for Python 3.13)
- **pandas**: Updated to `>=2.2.3` (has pre-built wheels for Python 3.13)
- **langchain**: Updated to `>=0.3.0` (Python 3.13 support)
- **faiss-cpu**: Updated to `>=1.13.1` (Python 3.13 support)
- **All other packages**: Updated to latest compatible versions

### 2. Code Updates

- **LangChain imports**: Updated to use `langchain_core` and `langchain_text_splitters` for version 0.3+
- **Fallback imports**: Added fallback imports for older LangChain versions
- **Import compatibility**: Code now works with both old and new LangChain versions

### 3. Installation Helper

Created `install_requirements.py` script that:
- Installs packages in the correct order
- Handles errors gracefully
- Provides better feedback during installation

## Installation Instructions

### Recommended Method (Python 3.13)

```bash
# Use the helper script
python install_requirements.py
```

### Alternative Method

```bash
# Install core packages first
pip install numpy>=2.0.0 pandas>=2.2.3

# Then install everything else
pip install -r requirements.txt
```

### If You Encounter Issues

1. **Pandas build errors**: Install pandas separately first
   ```bash
   pip install pandas>=2.2.3 --upgrade
   ```

2. **Numpy compatibility**: Ensure numpy 2.0+
   ```bash
   pip install numpy>=2.0.0 --upgrade
   ```

3. **Individual package installation**: Install problematic packages one by one
   ```bash
   pip install <package-name>
   ```

## Known Issues and Solutions

### Issue: pandas requires Visual Studio Build Tools

**Solution**: Use pre-built wheels (pandas>=2.2.3 has wheels for Python 3.13)

### Issue: Some packages don't have Python 3.13 wheels yet

**Solution**: 
- Check package documentation for Python 3.13 support
- Consider using Python 3.11 or 3.12 if critical packages don't support 3.13 yet
- Report issues to package maintainers

### Issue: torch installation is large

**Solution**: 
- For CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Or install without torch if not using NLI models: Comment out torch in requirements.txt

## Testing

After installation, verify everything works:

```bash
python validate_setup.py
python quick_start.py
```

## Compatibility Matrix

| Package | Minimum Version | Python 3.13 Support |
|---------|----------------|---------------------|
| numpy | 2.0.0 | ✅ Yes |
| pandas | 2.2.3 | ✅ Yes |
| langchain | 0.3.0 | ✅ Yes |
| faiss-cpu | 1.13.1 | ✅ Yes |
| sentence-transformers | 2.7.0 | ✅ Yes |
| streamlit | 1.39.0 | ✅ Yes |

## Fallback Options

If you continue to have issues with Python 3.13:

1. **Use Python 3.11 or 3.12**: These versions have better package support
2. **Use conda**: Conda may have better pre-built packages
3. **Use Docker**: Containerized environment with pre-configured Python version

## Support

If you encounter package-specific issues:
1. Check the package's GitHub issues for Python 3.13 support
2. Try installing the package individually to isolate the problem
3. Consider using a virtual environment with Python 3.11 or 3.12 as a temporary workaround

