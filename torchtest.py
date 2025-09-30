import os

print("VENV:", os.environ.get("VIRTUAL_ENV"))
print("PYTHONNOUSERSITE:", os.environ.get("PYTHONNOUSERSITE"))
print("LD_LIBRARY_PATH head:", os.environ.get("LD_LIBRARY_PATH","").split(":")[:3])

import torch

print(torch.version.cuda)
print(torch.__version__)