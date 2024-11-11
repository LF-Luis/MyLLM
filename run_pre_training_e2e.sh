#!/bin/bash
set -e

# Script to run all pre-training end-to-end.
# This script will first download training and validation data
# and then start running the pre-training loop.

export PYTHONPATH="${PYTHONPATH}:$PWD/src"

if [[ "$(basename "$PWD")" != "MyLLM" ]]; then
    echo "Error: You are not in the 'MyLLM' project directory. Current directory is '$PWD'."
    exit 1
fi

# Create log file for script, training logic handles its own logging
LOGFILE="$PWD/temp_data/logs/run_training_e2e_$(date +'%Y_%m_%d-%H_%M_%Z').log"
mkdir -p "$(dirname "$LOGFILE")" && [ -f "$LOGFILE" ] || touch "$LOGFILE"
: > "$LOGFILE" # Empty the log file
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$timestamp $*" | tee -a "$LOGFILE"
}

log "LOGFILE: $LOGFILE"
log "MAC_DEBUG is set to $MAC_DEBUG"

# Check if running on GPU-instance
if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    log "Running on NVIDIA GPU-instance!"
    pip install transformers && pip install tiktoken && pip install --upgrade scipy && pip install --upgrade networkx && pip install datasets && pip install tqdm
fi

# Download pretraining data
if [ ! -d "$PWD/temp_data/edu_fineweb10B" ]; then
    log "Downloading Pre-Training dataset."
    python src/data_processing/data_downloader.py
    log "Done downloading Pre-Training dataset."
else
    log "Skipping downloading Pre-Training dataset."
fi

# Start pretraining
log "Start training MyLLM."
python pretrain.py
log "Done training MyLLM."
