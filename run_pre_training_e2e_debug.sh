#!/bin/bash
set -e

# Set true to debug on a Mac (or any other system). This will download
# a very small subset of the data and train with smaller batches for less time.
# This is mainly to quckly see if the code is working e2e.
export DEBUG_MODE=True

./run_pre_training_e2e.sh
