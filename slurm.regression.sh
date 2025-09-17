#!/bin/bash
#SBATCH --job-name=regression_tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=30G
#SBATCH --output=%x-%j.out  # Default stdout file based on job name and job ID
#SBATCH --error=%x-%j.err   # Default stderr file based on job name and job ID
#SBATCH --time=00-04:00 # Time (DD-HH:MM)
#SBATCH --export=ALL # Added, propagates env vars for slurm.regression.sh
#SBATCH --mail-user==mark.spahl@mail.utoronto.ca
#SBATCH --mail-type=ALL


set -euo pipefail # Added Safety

# --- Optional: site-specific environment setup ---
module purge
module load cudacore/.12.6.2
module load cudnn/9.5.1.17
module load nccl/2.26.2 
module load python/3.12
# Activate venv
activate () {
	. venv/bin/activate
}
activate
# -----------------------------------------------

# Set cudnn path
export CUDNN_PATH="$(dirname "$EBROOTCUDNN")"

# Deterministic base for MASTER_PORT (avoids clashes across users) + check availability on rank-0
BASE=15000
SPAN=20000
CANDIDATE=$((BASE + (SLURM_JOB_ID % SPAN)))

choose_port() {
  # try to bind the candidate port; if busy, increment until free
  local p="${1:-29500}"
  while true; do
    # Try to bind using Python (immediately closes on success)
    python - <<'PY' "$p"
import socket, sys
p = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("", p))
    s.close()
    sys.exit(0)   # success
except OSError:
    sys.exit(1)        # in use -> nonzero exit to loop again
PY
    if [ $? -eq 0 ]; then
      echo "$p"
      return 0
    fi
    p=$((p+1))
    [ $p -ge 65000 ] && { echo "No free port found" >&2; return 1; }
  done
}
MASTER_PORT=$(choose_port "$CANDIDATE") || exit 1
export MASTER_PORT

echo "LOG_DIR_1: $LOG_DIR_1"
echo "LOG_DIR_2: $LOG_DIR_2"
echo "LOG_DIR_3: $LOG_DIR_3"
echo "CONFIG_1: $CONFIG_1"
echo "CONFIG_2: $CONFIG_2"
echo "CONFIG_3: $CONFIG_3"
echo "DATA_CONFIG: $DATA_CONFIG"
echo "CHECKPOINT_RELOAD_TEST: $CHECKPOINT_RELOAD_TEST"
echo "CUDNN_PATH: $CUDNN_PATH"
echo "MASTER_PORT: $MASTER_PORT"

echo "Running first job with config: ${LOG_DIR_1}/$(basename $CONFIG_1)"
srun --output="${LOG_DIR_1}/slurm-%j.out" --error="${LOG_DIR_1}/slurm-%j.err" \
    python launch.py train.py ${DATA_CONFIG} "${LOG_DIR_1}/$(basename $CONFIG_1)" && \
    echo "First job completed, check logs: ${LOG_DIR_1}/slurm-%j.{out,err}" || echo "First job failed"

echo "Running second job with config: ${LOG_DIR_2}/$(basename $CONFIG_2)"
srun --output="${LOG_DIR_2}/slurm-%j.out" --error="${LOG_DIR_2}/slurm-%j.err" \
    python launch.py train.py ${DATA_CONFIG} "${LOG_DIR_2}/$(basename $CONFIG_2)" && \
    echo "Second job completed, check logs: ${LOG_DIR_2}/slurm-%j.{out,err}" || echo "Second job failed"

if [ "$CHECKPOINT_RELOAD_TEST" = true ] ; then
    echo "Running checkpoint reload test with config: ${LOG_DIR_3}/$(basename $CONFIG_3)"
    srun --output="${LOG_DIR_3}/slurm-%j.out" --error="${LOG_DIR_3}/slurm-%j.err" \
        python launch.py train.py ${DATA_CONFIG} "${LOG_DIR_3}/$(basename $CONFIG_3)" && \
        echo "Checkpoint reload test completed, check logs: ${LOG_DIR_3}/slurm-%j.{out,err}" || echo "Checkpoint reload test failed"
fi
