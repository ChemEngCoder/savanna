#!/bin/bash
activate () {
	. venv/bin/activate
}

activate

python -c "import deepspeed; print(deepspeed.__version__)"