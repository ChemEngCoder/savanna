#!/bin/bash

module purge
module load python/3.12
module load cudacore/.12.6.2
module load cudnn/9.5.1.17
module load nccl/2.26.2
export CUDNN_PATH=$(dirname "$EBROOTCUDNN")/9.5.1.17