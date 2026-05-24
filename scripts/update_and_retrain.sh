#!/bin/bash
# Auto-update wrapper: fetch MISO data + retrain LSTM
# Called by systemd timer or crontab daily

set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Activate venv if exists
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

export GRIDSTATUS_ENABLED=1

python scripts/update_and_retrain.py --cron >> logs/update.log 2>&1
