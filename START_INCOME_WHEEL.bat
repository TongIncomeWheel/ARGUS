@echo off
echo ========================================
echo   INCOME WHEEL APP - Launcher
echo ========================================
echo.
echo   Select environment to launch:
echo.
echo   [1] PROD  (port 8501)
echo   [2] DEV   (port 8502)
echo.

set /p CHOICE="Enter 1 or 2: "

if "%CHOICE%"=="1" (
    set ENV_NAME=PROD
    set PORT=8501
) else if "%CHOICE%"=="2" (
    set ENV_NAME=DEV
    set PORT=8502
) else (
    echo.
    echo   Invalid choice. Please enter 1 or 2.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Launching %ENV_NAME% on port %PORT%...
echo ========================================
echo.

REM Change to Income Wheel directory
cd /d C:\Users\ashtz\IncomeWheel

echo Checking if port %PORT% is available...
powershell -Command "$conn = netstat -ano | findstr ':%PORT%' | findstr 'LISTENING'; if ($conn) { $procId = ($conn -split '\s+')[-1]; Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue; Write-Output 'Killed process on port %PORT%'; Start-Sleep -Seconds 2 }"

echo.
echo Starting Streamlit on port %PORT%...
echo.
echo ========================================
echo   App will be available at:
echo   http://localhost:%PORT%
echo ========================================
echo.
echo Press Ctrl+C to stop the app
echo.

python -m streamlit run app.py --server.port %PORT% --server.headless true

pause
