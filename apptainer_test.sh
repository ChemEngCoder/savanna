

export SCRATCH=$PWD
export CONTAINER="../pytorch_24.09-py3.12-mpi.sif"

module load openmpi/4.1.5

apptainer exec --nv --cleanenv \
  --bind $SCRATCH:$SCRATCH \
  $CONTAINER bash -lc '

    ls
    . venv/bin/activate

    echo $PWD

    unset PYTHONPATH
    export PYTHONNOUSERSITE=1

    # 2) compute venv site-packages and torch/lib WITHOUT importing torch
    # ★ put the venv’s Torch .so directory FIRST for the dynamic linker
    TORCH_LIB="$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib"
    echo $TORCH_LIB

    # 3) locate CUDA + MPI runtime dirs (best-effort, adjust as needed)
    CUDA_LIB="${CUDA_HOME:-/usr/local/cuda}/lib64"
    # OpenMPI example; fall back to common prefixes if ompi_info is unavailable
    OMPI_LIBDIR=$( (ompi_info --path libdir 2>/dev/null | awk "{print \$2}") || echo "" )
    [ -z "$OMPI_LIBDIR" ] && for d in /usr/lib/x86_64-linux-gnu /usr/lib /opt/openmpi*/lib; do
    [ -e "$d/libmpi.so" ] && OMPI_LIBDIR="$d" && break
    done

    # 4) build a minimal LD_LIBRARY_PATH (left-to-right search order)
    LD_NEW=""
    [ -d "$TORCH_LIB" ] && LD_NEW="$TORCH_LIB"
    [ -d "$CUDA_LIB"  ] && LD_NEW="${LD_NEW:+$LD_NEW:}$CUDA_LIB"
    [ -n "$OMPI_LIBDIR" ] && LD_NEW="${LD_NEW:+$LD_NEW:}$OMPI_LIBDIR"

    # 5) optionally append any existing entries after filtering out stale torch paths
    LD_OLD_FILTERED=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | \
    grep -vE "python3\.10/.*/torch(_tensorrt)?/lib" | paste -sd: -)
    export LD_LIBRARY_PATH="${LD_NEW}${LD_OLD_FILTERED:+:$LD_OLD_FILTERED}"

    # (optional) quick print
    echo "LD_LIBRARY_PATH head:"
    echo "$LD_LIBRARY_PATH" | tr ':' '\n' | head -n 5

    ls "$VIRTUAL_ENV/python3.12/site-packages/torch/lib"
    ls /usr/local/lib/python3.10/dist-packages/torch/lib
    echo $CUDA_HOME
    echo $CUDNN_PATH
    nvcc --version
    python --version
    python torchtest.py
'