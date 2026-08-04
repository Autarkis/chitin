@echo off
setlocal
cd /d "%~dp0"

if /I "%~1"=="--help" goto :help

powershell.exe -NoProfile -Command "try { $page = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:4179/'; if ($page.Content -like '*<title>Chitin Collider Lab</title>*') { exit 0 } } catch {}; exit 1" >nul 2>&1
if not errorlevel 1 (
  echo Chitin Collider Lab is already running. Opening it now...
  start "" "http://127.0.0.1:4179/"
  exit /b 0
)

where npm.cmd >nul 2>&1
if errorlevel 1 (
  echo.
  echo Chitin Collider Lab needs Node.js and npm.
  echo Install the current Node.js LTS release, then run START_DEMO.cmd again.
  echo.
  pause
  exit /b 1
)

echo Preparing Chitin Collider Lab...
call :build_package "..\..\integrations\wasm-lite"
if errorlevel 1 goto :failed
call :build_package "..\..\integrations\web"
if errorlevel 1 goto :failed

if not exist "node_modules\vite\bin\vite.js" (
  echo Installing demo dependencies...
  call npm install
  if errorlevel 1 goto :failed
)

echo.
echo Opening http://127.0.0.1:4179/
echo Keep this window open. Press Ctrl+C here to stop the demo.
echo.
call npm start
exit /b %errorlevel%

:build_package
pushd "%~1"
if not exist "node_modules\.bin\tsc.cmd" (
  echo Installing dependencies for %~1...
  call npm install
  if errorlevel 1 (
    popd
    exit /b 1
  )
)
echo Building %~1...
call npm run build
set "BUILD_RESULT=%errorlevel%"
popd
exit /b %BUILD_RESULT%

:failed
echo.
echo The demo could not be prepared. Review the error above, then try again.
echo.
pause
exit /b 1

:help
echo START_DEMO.cmd
echo Builds the local Chitin browser packages, starts Collider Lab on port 4179,
echo and opens it in the default browser. Keep the terminal open while using it.
exit /b 0
