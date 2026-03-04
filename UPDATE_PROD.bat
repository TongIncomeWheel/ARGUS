@echo off
echo ========================================
echo   ARGUS - Update Prod from GitHub
echo ========================================
echo.

cd /d C:\Users\ashtz\ARGUS_Prod

echo Pulling latest from GitHub main...
git pull origin main

echo.
echo ========================================
echo   ARGUS_Prod is now up to date.
echo ========================================
echo.
pause
