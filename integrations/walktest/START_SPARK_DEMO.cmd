@echo off
setlocal
cd /d "%~dp0"

if /I "%~1"=="--help" goto :help

powershell.exe -NoProfile -Command "try { $page = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:3219/spark-demo'; if ($page.Content -like '*<title>Chitin * Spark*Hull Shadows on Splats</title>*') { exit 0 } } catch {}; exit 1" >nul 2>&1
if not errorlevel 1 (
  echo Chitin Spark demo is already running. Opening it now...
  start "" "http://127.0.0.1:3219/spark-demo"
  exit /b 0
)

where npm.cmd >nul 2>&1
if errorlevel 1 (
  echo.
  echo The Chitin Spark demo needs Node.js and npm.
  echo Install the current Node.js LTS release, then run START_SPARK_DEMO.cmd again.
  echo.
  pause
  exit /b 1
)

if not exist "node_modules\serve\build\main.js" (
  echo Installing demo dependencies...
  call npm ci
  if errorlevel 1 goto :failed
)

echo Preparing the Chitin Spark demo...
call npm run prepare:release-candidate
if errorlevel 1 goto :failed
call npm run build:spark-demo
if errorlevel 1 goto :failed

echo.
echo Keep this window open. Press Ctrl+C here to stop the demo.
echo.
call npm run serve:spark
exit /b %errorlevel%

:failed
echo.
echo The demo could not be prepared. Review the error above, then try again.
echo.
pause
exit /b 1

:help
echo START_SPARK_DEMO.cmd
echo Builds the local Chitin browser packages, starts the Spark demo on port 3219,
echo and opens it in the default browser. Keep the terminal open while using it.
exit /b 0
