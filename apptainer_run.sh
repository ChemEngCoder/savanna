#!/bin/bash

export SAVANNA_DIR="${SAVANNA_DIR:-$PWD}"
IMAGE="containers/pytorch_24.09-py3.12-mpi.sif"
CONTAINER_VENV="venv/bin/activate"

apptainer exec --nv \
--bind "$SAVANNA_DIR:$SAVANNA_DIR" \
"$IMAGE" \
bash -lc "
    set -euo pipefail
    . \"$CONTAINER_VENV\"
    export TORCH_COMPILE_DISABLE=1
    ./setup_run_reload.sh
"