

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


    # Discover the venv’s site-packages robustly (no torch import needed)
    VENV_PURELIB=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    TORCH_LIB="$VENV_PURELIB/torch/lib"

    # Sanity: should exist and contain libtorch_python.so
    test -f "$TORCH_LIB/libtorch_python.so" || { echo "Torch lib dir not found: $TORCH_LIB"; exit 1; }

    # Build a minimal LD_LIBRARY_PATH (left→right search)
    CUDA_LIB="${CUDA_HOME:-/usr/local/cuda}/lib64"
    LD_NEW="$TORCH_LIB"
    [ -d "$CUDA_LIB" ] && LD_NEW="$LD_NEW:$CUDA_LIB"

    # (optional) add MPI runtime dir if you have it
    OMPI_LIBDIR=$( (ompi_info --path libdir 2>/dev/null | awk '{print $2}') || echo "" )
    [ -n "$OMPI_LIBDIR" ] && LD_NEW="$LD_NEW:$OMPI_LIBDIR"

    # Append any existing entries, but drop stale py3.10 torch paths
    LD_OLD_FILTERED=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
    | grep -vE 'python3\.10/.*/torch(_tensorrt)?/lib' | paste -sd: -)

    export LD_LIBRARY_PATH="${LD_NEW}${LD_OLD_FILTERED:+:$LD_OLD_FILTERED}"

    # Quick visibility
    echo "Using TORCH_LIB: $TORCH_LIB"
    echo "LD_LIBRARY_PATH (head):"; echo "$LD_LIBRARY_PATH" | tr ':' '\n' | head -n 8
    
    ls "$VIRTUAL_ENV/python3.12/site-packages/torch/lib"
    ls /usr/local/lib/python3.10/dist-packages/torch/lib
    echo $CUDA_HOME
    echo $CUDNN_PATH
    nvcc --version
    python --version
    python torchtest.py
'