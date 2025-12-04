#!/bin/bash
#SBATCH --time=00:02:00

activate () {
	. venv/bin/activate
}

activate

python filetest.py