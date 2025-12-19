@echo off
REM Setup script for Windows

echo ==========================================
echo Setting up Agentic AI Interface
echo ==========================================

REM Create directories
echo Creating directories...
if not exist papers mkdir papers
if not exist pdfs mkdir pdfs
if not exist monitoring\grafana\dashboards mkdir monitoring\grafana\dashboards

REM Check for .env file
if not exist .env (
    echo Creating .env file from .env.example...
    if exist .env.example (
        copy .env.example .env
        echo Please edit .env file with your API keys
    ) else (
        echo Warning: .env.example not found
    )
) else (
    echo .env file already exists
)

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Setup monitoring
echo Setting up monitoring...
python monitoring_setup.py

echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Run: python collect_papers.py
echo 3. Run: python preprocessing_notebook.py
echo 4. Run: streamlit run app.py
echo.

pause

