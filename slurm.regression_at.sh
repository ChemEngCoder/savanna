#!/bin/bash
#SBATCH --job-name=regression_tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=30G
#SBATCH --output=%x-%j.out          # stdout for this wrapper script
#SBATCH --error=%x-%j.err           # stderr for this wrapper script
#SBATCH --time=00-04:00             # Time (DD-HH:MM)
#SBATCH --export=ALL                # propagate env vars from run_regression_test.sh
#SBATCH --mail-user=mark.spahl@mail.utoronto.ca
#SBATCH --mail-type=ALL

set -euo pipefail

# ------------------------------
# Site-specific environment
# ------------------------------
module purge
module load apptainer/1.3.5     # or singularity/apptainer module name on your site
module load cudacore/.12.6.2
module load cudnn/9.5.1.17
module load nccl/2.26.2

# If your code expects this:
export CUDNN_PATH="$(dirname "$EBROOTCUDNN")/9.5.1.17"

# ------------------------------
# Paths & container settings
# ------------------------------

# Use existing SCRATCH if defined, else current dir
export SCRATCH="${SCRATCH:-../}"

# Location of your Savanna repo inside SCRATCH
export SAVANNA_DIR="$SCRATCH/savanna"

# Apptainer image + venv inside the image
IMAGE="pytorch_24.09-py3.13-mpi.sif"
CONTAINER_VENV="environments/venv_cuda12.6_at/bin/activate"

echo "Host:            $(hostname)"
echo "SCRATCH:         $SCRATCH"
echo "SAVANNA_DIR:     $SAVANNA_DIR"
echo "IMAGE:           $IMAGE"
echo "CONTAINER_VENV:  $CONTAINER_VENV"
echo "LOG_DIR_1:       ${LOG_DIR_1:-<unset>}"
echo "LOG_DIR_2:       ${LOG_DIR_2:-<unset>}"
echo "LOG_DIR_3:       ${LOG_DIR_3:-<unset>}"
echo "CONFIG_1:        ${CONFIG_1:-<unset>}"
echo "CONFIG_2:        ${CONFIG_2:-<unset>}"
echo "CONFIG_3:        ${CONFIG_3:-<unset>}"
echo "DATA_CONFIG:     ${DATA_CONFIG:-<unset>}"
echo "CHECKPOINT_RELOAD_TEST: ${CHECKPOINT_RELOAD_TEST:-<unset>}"
echo "CUDNN_PATH:      ${CUDNN_PATH:-<unset>}"
echo "MASTER_PORT:     ${MASTER_PORT:-<unset>}"
echo

# ------------------------------
# Helper: run a single regression inside the container
# ------------------------------

run_in_container() {
    local tag="$1"        # e.g. reg1 / reg2 / reg3
    local cfg_path="$2"   # full path to updated config (in LOG_DIR_X)
    local log_dir="$3"    # the corresponding log dir (LOG_DIR_X)

    local out_file="${log_dir}/slurm-${SLURM_JOB_ID}-${tag}.out"
    local err_file="${log_dir}/slurm-${SLURM_JOB_ID}-${tag}.err"

    echo "[$tag] Running with config: $cfg_path"
    echo "[$tag] Logs: $out_file / $err_file"
    echo

    srun \
      --ntasks=1 \
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" \
      --gpus=1 \
      --output="$out_file" \
      --error="$err_file" \
      apptainer exec --nv \
        --bind "$SCRATCH:$SCRATCH" \
        "$IMAGE" \
        bash -lc "
          set -euo pipefail
          . \"$CONTAINER_VENV\"
          cd \"$SAVANNA_DIR\"
          export TORCH_COMPILE_DISABLE=1
          python launch.py train.py \"$DATA_CONFIG\" \"$cfg_path\"
        "
}

# ------------------------------
# Compute paths to UPDATED configs
# (these are created by run_regression_test.sh)
# ------------------------------

CFG_1="${LOG_DIR_1}/$(basename "$CONFIG_1")"
CFG_2="${LOG_DIR_2}/$(basename "$CONFIG_2")"
CFG_3="${LOG_DIR_3}/$(basename "$CONFIG_3")"

# ------------------------------
# Run regression tests
# ------------------------------

# 1) First regression
run_in_container "reg1" "$CFG_1" "$LOG_DIR_1" || echo "[reg1] FAILED"

# 2) Second regression
run_in_container "reg2" "$CFG_2" "$LOG_DIR_2" || echo "[reg2] FAILED"

# 3) Optional checkpoint reload test
if [ "${CHECKPOINT_RELOAD_TEST}" = true ] ; then
    run_in_container "reg3" "$CFG_3" "$LOG_DIR_3" || echo "[reg3] FAILED"
else
    echo "Checkpoint reload test disabled (CHECKPOINT_RELOAD_TEST != true)"
fi

echo "All scheduled regression runs finished."