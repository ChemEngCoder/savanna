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

choose_port_2() {
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
    print(p)           # success -> print the usable port
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

MASTER_PORT_1=$(choose_port "$CANDIDATE") || exit 1
MASTER_PORT_2=$(choose_port_2 "$CANDIDATE") || exit 1
echo "MASTER_PORT_1: $MASTER_PORT_1"
echo "MASTER_PORT_2: $MASTER_PORT_2"