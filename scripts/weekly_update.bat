@echo off
REM Weekly Data Update & Sync to Server
REM Schedule: Windows Task Scheduler — run weekly (e.g., every Monday 8:00 AM)
REM
REM To schedule:
REM   1. Press Win+R, type taskschd.msc
REM   2. Create Basic Task → Name: "EPF Weekly Update"
REM   3. Trigger: Weekly, Monday, 8:00 AM
REM   4. Action: Start a program → Browse to this .bat file
REM   5. Finish

REM Project root: defaults to this script's parent folder.
REM Override with the EPF_PROJECT_DIR environment variable if needed.
if "%EPF_PROJECT_DIR%"=="" (
    set PROJECT_DIR=%~dp0..
) else (
    set PROJECT_DIR=%EPF_PROJECT_DIR%
)
set LOG_FILE=%PROJECT_DIR%\logs\weekly_update.log

echo ============================================ >> %LOG_FILE%
echo Update started at %date% %time% >> %LOG_FILE%

cd /d %PROJECT_DIR%

REM Step 1: Fetch new MISO data and retrain LSTM
echo [1/3] Fetching MISO data + retraining LSTM... >> %LOG_FILE%
python scripts/update_and_retrain.py >> %LOG_FILE% 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] update_and_retrain.py failed >> %LOG_FILE%
    exit /b 1
)

REM Step 2: Run RAG evaluation (optional, for monitoring)
echo [2/3] Running RAG evaluation... >> %LOG_FILE%
python scripts/evaluate_rag.py >> %LOG_FILE% 2>&1

REM Step 3: Evaluate LSTM model quality
echo [3/3] Running LSTM evaluation... >> %LOG_FILE%
python scripts/evaluate_lstm.py >> %LOG_FILE% 2>&1

REM Step 4: Sync to cloud server
echo [4/4] Syncing to server... >> %LOG_FILE%
python scripts/sync_to_server.py >> %LOG_FILE% 2>&1

echo Update completed at %date% %time% >> %LOG_FILE%
echo ============================================ >> %LOG_FILE%
