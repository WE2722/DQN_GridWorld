@echo off
REM run_app.bat - DQN GridWorld Streamlit App Launcher

echo.
echo ================================================
echo DQN GridWorld Streamlit App Launcher
echo ================================================
echo.

set SCRIPT_DIR=%~dp0
set APP_FILE=%SCRIPT_DIR%app.py
set PORT=8501

REM Parse port argument
if /i "%~1"=="--port" set PORT=%~2
if /i "%~1"=="--help" (
    echo Usage: run_app.bat [--port PORT]
    echo   --port PORT   Specify port default: 8501
    pause
    exit /b 0
)

REM Check if app.py exists
echo Checking for app.py...
if not exist "%APP_FILE%" (
    echo ERROR: app.py not found in %SCRIPT_DIR%
    pause
    exit /b 1
)
echo [OK] app.py found

REM Use hardcoded path to parent venv
echo.
echo Checking Python environment...
set PYTHON_EXE=C:\Users\Wiam\Desktop\Livrables Masrour\.venv\Scripts\python.exe

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
