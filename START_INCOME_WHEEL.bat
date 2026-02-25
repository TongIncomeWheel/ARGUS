@echo off
echo ========================================
echo   INCOME WHEEL APP - Starting...
echo ========================================
echo.

REM Change to Income Wheel directory
cd /d C:\Users\ashtz\IncomeWheel

echo Checking if port 8501 is available...
powershell -Command "$conn = netstat -ano | findstr ':8501' | findstr 'LISTENING'; if ($conn) { $procId = ($conn -split '\s+')[-1]; Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue; Write-Output 'Killed process on port 8501'; Start-Sleep -Seconds 2 }"

echo.
echo Starting Streamlit on port 8501...
echo.
echo ========================================
echo   App will be available at:
echo   http://localhost:8501
echo ========================================
echo.
echo Press Ctrl+C to stop the app
echo.

python -m streamlit run app.py --server.port 8501 --server.headless true

pause
