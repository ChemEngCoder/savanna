#!/bin/bash

export TS="$(date +%Y%m%d%H%M)"
export DATA_CONFIG="configs/data/opengenome.yml"
export CONFIG_1="configs/test/regression_1.yml"
export LOG_DIR_1="$(pwd)/logs/regression/${TS}regression_1"
mkdir -p "$LOG_DIR_1"
export CHECKPOINT_DIR="$(pwd)/logs/regression/checkpoints"
export UPDATED_CONFIG="${LOG_DIR_1}/$(basename ${CONFIG_1})"
cp "${CONFIG_1}" "$UPDATED_CONFIG"
python launch.py train.py ${DATA_CONFIG} "${LOG_DIR_1}/$(basename $CONFIG_1)"
