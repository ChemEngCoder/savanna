

export SCRATCH=$PWD
export CONTAINER="../pytorch_24.09-py3.12-mpi.sif"

module load openmpi/4.1.5

apptainer exec --nv \
  --bind $SCRATCH:$SCRATCH \
  $CONTAINER bash -lc '

    ls
    . venv/bin/activate

    echo $PWD

    unset PYTHONPATH
    export PYTHONNOUSERSITE=1

    export LD_LIBRARY_PATH="$VIRTUAL_ENV/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"


    echo $CUDA_HOME
    echo $CUDNN_PATH
    nvcc --version
    python --version
    python torchtest.py
