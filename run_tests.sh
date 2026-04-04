#!/usr/bin/env bash
# run_tests.sh  –  Run smoke tests inside the smart_fund_advisor conda env

set -e
PROJ="/Users/chaitanya/Downloads/Submission/Code/20Feb26"
PYTHON="$HOME/miniconda3/envs/smart_fund_advisor/bin/python"

# Fallback: try common conda install locations
if [ ! -f "$PYTHON" ]; then
    PYTHON="$HOME/anaconda3/envs/smart_fund_advisor/bin/python"
fi
if [ ! -f "$PYTHON" ]; then
    PYTHON="$(conda run -n smart_fund_advisor which python)"
fi

echo "Using Python: $PYTHON"
exec "$PYTHON" "$PROJ/test_system.py"
