@echo off
echo ==========================================
echo Plant Disease Detection - Streamlit App
echo ==========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Starting Streamlit app...
echo App will be available at: http://localhost:8501
echo.

REM Run the app
python -m streamlit run app.py

REM Deactivate on exit
call venv\Scripts\deactivate
pause
