import os
import deepspeed

ds_test_setting = os.environ['DS_TEST']

if ds_test_setting == "nccl":
    deep