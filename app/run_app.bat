@echo off
echo ==========================================
echo   Plant Disease Detection - Streamlit App
echo ==========================================
echo.

REM Get the directory where this batch file is located
cd /d "%~dp0"

REM Check if virtual environment exists
if exist "venv_streamlit\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv_streamlit\Scripts\activate
) else (
    echo [WARNING] Virtual environment not found.
    echo [INFO] Please run: python -m venv venv_streamlit
    echo [INFO] Then: pip install -r requirements_streamlit.txt
    echo.
    pause
    exit /b 1
)

echo [INFO] Starting Streamlit application...
echo [INFO] The app will open in your browser at http://localhost:8501
echo.

REM Run the Streamlit application
streamlit run app.py

REM Pause to see any errors
if errorlevel 1 (
    echo.
    echo [ERROR] Application failed to start.
    pause
)
