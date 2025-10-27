@echo off
REM Desktop launcher for DQN GridWorld Visualizer
REM Double-click this file to launch the app

echo.
echo ================================================
echo DQN GridWorld Streamlit App Launcher
echo ================================================
echo.

cd /d "%~dp0"

set APP_FILE=%~dp0app.py
set PORT=8501
set PYTHON_EXE=C:\Users\Wiam\Desktop\Livrables Masrour\.venv\Scripts\python.exe

REM Check if app.py exists
echo Checking for app.py...
if not exist "%APP_FILE%" (
    echo ERROR: app.py not found
    pause
    exit /b 1
)
echo [OK] app.py found

REM Check Python environment
echo.
echo Checking Python environment...
if exist "%PYTHON_EXE%" (
    echo [OK] Using parent virtual environment Python
) else (
    echo ERROR: Python not found at %PYTHON_EXE%
    pause
    exit /b 1
)

echo.
echo Starting Streamlit app on port %PORT%...
echo Browser will open automatically at: http://localhost:%PORT%
echo.
echo Press Ctrl+C to stop the server.
echo.

REM Start the app - Streamlit will automatically open the browser
"%PYTHON_EXE%" -m streamlit run "%APP_FILE%" --server.port %PORT%

REM Only pause if there was an error
if errorlevel 1 (
    echo.
    echo Server stopped with an error.
    pause
)
