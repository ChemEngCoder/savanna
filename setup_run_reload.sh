#!/bin/bash

TS="$(date +%Y%m%d%H%M)"
TS="202512061119"

# Command-line arguments
DATA_CONFIG="configs/data/opengenome.yml"
CONFIG_1="configs/test/regression_1.yml"
CONFIG_2="configs/test/regression_2.yml"
CONFIG_3="configs/test/regression_3_checkpoint_reload.yml"
LOG_DIR="$(pwd)/logs/regression"
CHECKPOINT_DIR="${LOG_DIR}/checkpoints"
TRAIN_ITERS_1="2000"
TRAIN_ITERS_2="2000"
TRAIN_ITERS_3="50"
CHECKPOINT_RELOAD_TEST=true

# Update dirs
CHECKPOINT_DIR="${CHECKPOINT_DIR}/${TS}"
LOG_DIR="${LOG_DIR}/${TS}"

LOG_DIR_1="${LOG_DIR}/regression_1"
LOG_DIR_2="${LOG_DIR}/regression_2"
LOG_DIR_3="${LOG_DIR}/regression_3_checkpoint_reload"

UPDATED_CONFIG="${LOG_DIR_3}/$(basename ${CONFIG_3})"

python launch.py train.py ${DATA_CONFIG} ${UPDATED_CONFIG}
