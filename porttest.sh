#!/bin/bash
#SBATCH --time=00:02:00
# Deterministic base (avoids clashes across users) + check availability on rank-0
BASE=15000
SPAN=20000
CANDIDATE=$((BASE + (SLURM_JOB_ID % SPAN)))

choose_port() {
    local p="$1"
    while ss -lnt 2>/dev/null | awk '{print $4}' | grep -q ":$p$"; do
      p=$((p+1))
      [ $p -ge 65000 ] && { echo "No free port found" >&2; return 1; }
    done
    echo "$p"
}
MASTER_PORT=$(choose_port "$CANDIDATE") || exit 1
echo "MASTER_PORT: $MASTER_PORT"
export MASTER_PORT