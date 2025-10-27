@echo off
echo ===== Testing Batch File =====
echo.

set "PYTHON_EXE=C:\Users\Wiam\Desktop\Livrables Masrour\.venv\Scripts\python.exe"

echo Python path: %PYTHON_EXE%
echo.

if exist "%PYTHON_EXE%" (
    echo [OK] Python executable found!
    echo.
    echo Testing Python version:
    "%PYTHON_EXE%" --version
    echo.
    echo Testing torch import:
    "%PYTHON_EXE%" -c "import torch; print('torch version:', torch.__version__)"
    echo.
    echo Testing streamlit:
    "%PYTHON_EXE%" -m streamlit --version
    echo.
    echo [SUCCESS] All checks passed!
) else (
    echo [ERROR] Python not found at: %PYTHON_EXE%
)

echo.
pause
