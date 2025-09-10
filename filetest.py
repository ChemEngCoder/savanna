from os import listdir
from os.path import join

import numpy as np

from savanna.data.data_utils import build_the_dataset, get_normalized_weights_and_num_samples
from savanna.data.indexed_dataset import make_dataset as make_indexed_dataset

path = join("data", "evo", "pretraining_or_both", "gtdb_v220_imgpr")
files = [f for f in listdir(path)]
print(files)

train_path = "/data/evo/pretraining_or_both/gtdb_v220_imgpr/data_gtdb_train_chunk1_text_CharLevelTokenizer_document"
test_path = "/data/evo/pretraining_or_both/gtdb_v220_imgpr/data_gtdb_test_chunk1_text_CharLevelTokenizer_document"
valid_path = "/data/evo/pretraining_or_both/gtdb_v220_imgpr/data_gtdb_valid_chunk1_text_CharLevelTokenizer_document"

i = 0

train_iters = 2000
eval_interval = 100
eval_iters = (train_iters // eval_interval + 1) * 20
test_iters = 20

#train_batch_size
micro_batch = 16 #train_micro_batch_size_per_gpu,
grad_acc = 1 #gradient_accumulation_steps

#dp_world_size
global_num_gpus = 4
pp_size = 1 #pipe_parallel_size
mp_size = 1 #model_parallel_size
cp_size = 1#context_parallel_size
dp_world_size = (global_num_gpus / pp_size) / (mp_size * cp_size)

train_batch_size = micro_batch * grad_acc #train_batch_size
train_batch_size *= dp_world_size

per_ds_eval_iters = 0

train_val_test_num_samples = [
            train_iters * train_batch_size,
            eval_iters * train_batch_size,
            test_iters * train_batch_size,
            per_ds_eval_iters * train_batch_size
        ]

seq_length = 8192
seed = 1234
mmap_warmup: bool = False

train_data_weights = [1.]
valid_data_weights = [1.]
test_data_weights = [1.]

train_weights, train_num_samples = get_normalized_weights_and_num_samples(
                train_data_weights, train_val_test_num_samples[0]
            )
valid_weights, valid_num_samples = get_normalized_weights_and_num_samples(
    valid_data_weights, train_val_test_num_samples[1]
)
test_weights, test_num_samples = get_normalized_weights_and_num_samples(
    test_data_weights, train_val_test_num_samples[2]
)

train_dataset = build_the_dataset(
                    data_prefix=train_path,
                    name=f"train_{i}",
                    data_impl="mmap",
                    num_samples=train_num_samples[i],
                    seq_length=seq_length,
                    seed=seed,
                    skip_warmup=(not mmap_warmup),
                    build_index_mappings=True,
                    enforce_sample_length=False,
                    sample_dtype=np.int64,
                    global_config=None
)