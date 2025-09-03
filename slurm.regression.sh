#!/bin/bash
#SBATCH --job-name=regression_tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=30G
#SBATCH --output=%x-%j.out  # Default stdout file based on job name and job ID
#SBATCH --error=%x-%j.err   # Default stderr file based on job name and job ID
#SBATCH --time=00-04:00 # Time (DD-HH:MM)
#SBATCH --export=ALL # Added, propagates env vars for slurm.regression.sh
#SBATCH --mail-user==mark.spahl@mail.utoronto.ca
#SBATCH --mail-type=ALL


set -euo pipefail # Added Safety

# --- Optional: site-specific environment setup ---
# module purge
# module load cuda/12.1
# module load python/3.10
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate llm-train
# -----------------------------------------------

echo "LOG_DIR_1: $LOG_DIR_1"
echo "LOG_DIR_2: $LOG_DIR_2"
echo "LOG_DIR_3: $LOG_DIR_3"
echo "CONFIG_1: $CONFIG_1"
echo "CONFIG_2: $CONFIG_2"
echo "CONFIG_3: $CONFIG_3"
echo "DATA_CONFIG: $DATA_CONFIG"
echo "CHECKPOINT_RELOAD_TEST: $CHECKPOINT_RELOAD_TEST"

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
