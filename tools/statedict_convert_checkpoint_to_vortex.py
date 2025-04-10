"""
Checkpoint conversion code from savanna to vortex

python3 convert_checkpoint_to_vortex.py

@mp: This works with the latest version of Evo2 models (SHC, trained post onboarding on NVIDIA cluster). It does NOT work with SHC 1.5 models (e.g. VP run)

Also computes a dummy set of logits to test correctness of the conversion

Make sure to check whether the GatedMLP is using the right activation after layer idx 1 when computing logits. If the activation func is still a lambda, force it to be F.gelu
"""

import os
from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from savanna.ops.vandermonde import log_vandermonde_naive as log_vandermonde
from opt_einsum import contract
import glob
from collections import OrderedDict

from savanna.arguments import GlobalConfig
from savanna.model.backbone import ParallelBlockPipe, EmbeddingPipe, NormPipe, Lambda

from tools.load_checkpoint_from_deepspeed_raw import load_savanna_checkpoint

# import rearrange
from einops import rearrange


KEY_UPDATE_DICT_HYENA = {
    # hyena
    "mixer.mixer.filter.h": "filter.h",
    "mixer.mixer.filter.kernel.B": "filter.B",
    "mixer.mixer.filter.kernel.C": "filter.C",
    "mixer.mixer.conv_bias": "filter.D",
    "mixer.mixer.filter.decay": "",
    "mixer.mixer.filter.gamma": "",
    "mixer.mixer.filter.kernel.log_dt": "filter.log_dt",
    "mixer.mixer.filter.kernel.inv_A_real": "filter.inv_A_real",
    "mixer.mixer.filter.kernel.A_imag": "filter.A_imag",
    # short conv
    "mixer.hyena_proj_conv.short_conv_weight": "filter.short_filter_weight",
    "mixer.hyena_proj_conv.short_conv_bias": "filter.short_filter_bias",
    "mixer.mixer.short_conv_weight": "",
    "mixer.mixer.short_conv.short_conv_weight": "",
    # "mixer.hyena_proj_conv.weight": "filter.short_filter_weight",
    "mixer.hyena_proj_conv.bias": "filter.short_filter_bias",
    # rope
    "mixer.rotary_emb.inv_freq": "rotary_emb.inv_freq",
    # qkv proj
    "mixer.dense_projection.weight": "projections.weight",
    "mixer.dense_projection.bias": "projections.bias",
    "mixer.dense_projection._extra_state": "projections._extra_state",
    # mlp
    "mlp.w1.weight": "mlp.l1.weight",
    "mlp.w2.weight": "mlp.l2.weight",
    "mlp.w3.weight": "mlp.l3.weight",
    # dense layers
    "mixer.dense.weight": "out_filter_dense.weight",
    "mixer.dense.bias": "out_filter_dense.bias",
    # to scrap
    "mlp.w1._extra_state": "",
    "mlp.w2._extra_state": "",
    "mlp.w3._extra_state": "",
    "mixer.dense._extra_state": "",
    "post_attention_layernorm.scale": "",
    "outer_mlp_layernorm.scale": "",
    "input_layernorm.weight": "pre_norm.scale",
    "post_attention_layernorm.weight": "post_norm.scale",
    "input_layernorm.weight": "pre_norm.scale",
    "input_layernorm.scale": "pre_norm.scale",
    "pre_mlp_layernorm.scale": "post_norm.scale",
    #
    "input_layernorm.weight": "pre_norm.scale",
    "input_layernorm.scale": "pre_norm.scale",
    "pre_mlp_layernorm.scale": "post_norm.scale",
    "pre_mlp_layernorm.weight": "post_norm.scale",
    "outer_mlp_layernorm.weight": "",
    #
    "mixer.mixer.filter.act.freq": "",
    "mixer.mixer.filter.pos_emb.t": "",
    "mixer.mixer.filter.pos_emb.z": "",
    "mixer.mixer.filter.implicit_filter.0.weight": "",
    "mixer.mixer.filter.implicit_filter.0.bias": "",
    "mixer.mixer.filter.implicit_filter.1.freq": "",
    "mixer.mixer.filter.implicit_filter.2.weight": "",
    "mixer.mixer.filter.implicit_filter.2.bias": "",
    "mixer.mixer.filter.implicit_filter.3.freq": "",
    "mixer.mixer.filter.implicit_filter.4.weight": "",
    "mixer.mixer.filter.implicit_filter.4.bias": "",
    "mixer.mixer.filter.implicit_filter.5.freq": "",
    "mixer.mixer.filter.final_filter.weight": "",
    "mixer.mixer.filter.modulation.weight": "",
    #
    "mlp.w1.weight": "mlp.l1.weight",
    "mlp.w2.weight": "mlp.l2.weight",
    "mlp.w3.weight": "mlp.l3.weight",

    "norm.weight": "norm.scale",
}

KEY_UPDATE_DICT_ATTENTION = {
    "mixer.dense_projection.weight": "inner_mha_cls.Wqkv.weight",
    "mixer.dense_projection.bias": "inner_mha_cls.Wqkv.bias",
    "mixer.dense.weight": "inner_mha_cls.out_proj.weight",
    "mixer.dense.bias": "inner_mha_cls.out_proj.bias",
    "mixer.o_proj.weight": "inner_mha_cls.out_proj.weight",
    "mixer.o_proj.bias": "inner_mha_cls.out_proj.bias",
    # rope
    # "attention.rotary_emb.inv_freq": "inner_mha_cls.rotary_emb.inv_freq",
    "mixer.rotary_emb.inv_freq": "inner_mha_cls.rotary_emb.inv_freq",
    # to scrap
    "mlp.w1._extra_state": "",
    "mlp.w2._extra_state": "",
    "mlp.w3._extra_state": "",
    "attention.dense._extra_state": "",
    "post_attention_layernorm.scale": "",
    "outer_mlp_layernorm.weight": "",
    "outer_mlp_layernorm.scale": "",
    "mixer.dense_projection._extra_state": "",
    "mixer.q_proj.weight": "",
    "mixer.k_proj.weight": "",
    "mixer.v_proj.weight": "",
    # mlp
    "mlp.w1.weight": "mlp.l1.weight",
    "mlp.w2.weight": "mlp.l2.weight",
    "mlp.w3.weight": "mlp.l3.weight",
    #
    "input_layernorm.weight": "pre_norm.scale",
    "post_attention_layernorm.weight": "post_norm.scale",
    "input_layernorm.weight": "pre_norm.scale",
    "input_layernorm.scale": "pre_norm.scale",
    "pre_mlp_layernorm.scale": "post_norm.scale",
    "pre_mlp_layernorm.weight": "post_norm.scale",
    #
    "mlp.gate_proj.weight": "mlp.l1.weight",
    "mlp.up_proj.weight": "mlp.l2.weight",
    "mlp.down_proj.weight": "mlp.l3.weight",
    # misc
    "final_linear.weight": "word_embeddings.weight",
    "norm.weight": "norm.scale",
    #mlp
}

KEY_UPDATE_DICT_EMBEDDING = {
    "word_embeddings.weight": "embedding_layer.weight",
}

KEY_UPDATE_DICT_NORM = {
    "norm.scale": "norm.scale",
    "norm.weight": "norm.scale",
}

def remove_state_dict_prefixes(state_dict):
    for k in list(state_dict.keys()):
        if k.startswith("module."):
            state_dict[k[7:]] = state_dict.pop(k)
        elif k.startswith("sequential."):
            state_dict[k[10:]] = state_dict.pop(k)
    return state_dict


def detect_module_cls(state_dict_keys):
    keys = set(state_dict_keys)
    
    if any('word_embeddings.weight' in k for k in keys):
        return "embedding"
    
    if len(keys) == 1 and any('norm' in k for k in keys):
        return "norm"
        
    # Check for hyena variants
    if any('mixer.mixer.filter.p' in k for k in keys):
        return "hyena"
    # For blocks with filter.h, we need to distinguish medium vs short
    elif any('mixer.mixer.filter.h' in k for k in keys) and any('mixer.mixer.filter.decay' in k for k in keys):
        return "hyena_medium_conv"
    # Default remaining hyena blocks to short conv
    elif any('mixer.hyena_proj_conv.short_conv_weight' in k for k in keys):
        return "hyena_short_conv"
            
    # Check for attention keys
    if any('mixer.rotary_emb.inv_freq' in k for k in keys):
        return "attention"

    return "hyena"

def convert_module_state_dict(state_dict, operator_type, config):
    """Convert a pretrained savanna checkpoint state_dict to stripedhyena format"""
    print(f"Converting state dict for operator type: {operator_type}")
    
    if operator_type == "hyena":
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_HYENA
    elif operator_type == "hyena_medium_conv":
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_HYENA
    elif operator_type == "hyena_short_conv":
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_HYENA
    elif operator_type == "embedding":
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_EMBEDDING
    elif operator_type == "norm":
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_NORM
    else:
        KEY_UPDATE_DICT = KEY_UPDATE_DICT_ATTENTION

    new_state_dict = OrderedDict()

    # First pass - handle basic key conversions
    for k in state_dict.keys():
        # Skip p and R tensors as they'll be handled separately
        if k in ["mixer.mixer.filter.p", "mixer.mixer.filter.R"]:
            continue
            
        new_k = KEY_UPDATE_DICT.get(k, k)
        if new_k != "":
            if "_extra_state" in new_k:
                new_state_dict[new_k] = state_dict[k]
            elif hasattr(state_dict[k], 'shape'):
                if "filter.short_filter_weight" in new_k:
                    new_state_dict[new_k] = state_dict[k][:,None]
                else:
                    new_state_dict[new_k] = state_dict[k]  # Keep original dtype

    # Second pass - handle special hyena conversions
    if operator_type == "hyena":
        if "mixer.mixer.filter.p" in state_dict and "mixer.mixer.filter.R" in state_dict:
            # First convert all inputs to float32
            p = state_dict["mixer.mixer.filter.p"].to(torch.float32)
            gamma = state_dict["mixer.mixer.filter.gamma"].to(torch.float32)
            
            # Then do reshape
            p = p.reshape(config.num_groups_hyena, config.hyena_filter_order)
            
            # Compute exp(gamma) first
            exp_gamma = torch.exp(gamma)
            
            # Then compute the poles
            logp = -torch.exp(p)
            logp = logp * exp_gamma
            
            # Finally add the extra dimension
            logp = logp[..., None]
            
            # Store result
            new_state_dict["filter.log_poles"] = logp
            new_state_dict["filter.residues"] = state_dict["mixer.mixer.filter.R"].to(torch.float32).reshape(
                config.num_groups_hyena, config.hyena_filter_order)
            
    elif operator_type == "hyena_medium_conv":
        if "mixer.mixer.filter.h" in state_dict:
            h = state_dict["mixer.mixer.filter.h"]
            decay = state_dict["mixer.mixer.filter.decay"]
            L = config.hyena_medium_conv_len if hasattr(config, 'hyena_medium_conv_len') else h.shape[1]
            h = h[:, :L] * decay[:, :L]
            new_state_dict["filter.h"] = h.unsqueeze(1)
    
    elif operator_type == "hyena_short_conv":
        if "mixer.mixer.short_conv.short_conv_weight" in state_dict:
            h = state_dict["mixer.mixer.short_conv.short_conv_weight"]
            new_state_dict["filter.h"] = h

    return new_state_dict

from collections import defaultdict
def organize_blocks(module_dict):
    # First, group all keys by their block number but keep full names
    block_dicts = defaultdict(dict)
    # Process all keys
    for key in module_dict.keys():
        if key.startswith('sequential.'):
            parts = key.split('.')
            block_idx = int(parts[1])
            block_dicts[key] = module_dict[key]  # Keep the full key
    return dict(sorted(block_dicts.items()))

def main():
    # config_path = '/scratch/hielab/gbrixi/evo2/evo2_chimera/1b_stripedhyena2_2M_lr0.00015/global_step490000/configs/1b_stripedhyena2_base_2M.yml'
    # new_checkpoint_path = '/scratch/hielab/gbrixi/evo2/vortex_interleaved/1b_testing/'
    # checkpoint_path = '/scratch/hielab/gbrixi/evo2/evo2_interleaved/1b/global_step490000/mp_rank_00_model_states.pt'
    # iteration = 490_000

    # config_path = '/scratch/hielab/gbrixi/evo2/evo2_interleaved/7b/global_step500000/7b_stripedhyena2_base_4M_resume.yml'
    # new_checkpoint_path = '/scratch/hielab/gbrixi/evo2/vortex_interleaved/7b_testing/'
    # checkpoint_path = '/scratch/hielab/gbrixi/evo2/evo2_interleaved/7b/global_step500000/mp_rank_00_model_states.pt'
    # iteration = 500_000

    # config_path = '/large_storage/hielab/gbrixi/checkpoints/evo2_interleaved/7b_ecoli/mp1/global_step4500/configs/7b-1M-ecoli-ft.yml'
    # new_checkpoint_path = '/scratch/hielab/gbrixi/evo2/vortex_interleaved/7b_1m_ecoli/'
    # checkpoint_path = '/large_storage/hielab/gbrixi/checkpoints/evo2_interleaved/7b_ecoli/mp1/global_step4500/mp_rank_00_model_states.pt'
    # iteration = 4500

    # config_path = '/large_storage/hielab/gbrixi/checkpoints/evo2_interleaved/7b_klebsiella/mp1/global_step4500/configs/7b-1M-klebsiella-ft.yml'
    # new_checkpoint_path = '/scratch/hielab/gbrixi/evo2/vortex_interleaved/7b_1m_klebsiella/'
    # checkpoint_path = '/large_storage/hielab/gbrixi/checkpoints/evo2_interleaved/7b_klebsiella/mp1/global_step4500/mp_rank_00_model_states.pt'
    # iteration = 4500
    
    # checkpoint_path = '/large_storage/hielab/brianhie/checkpoints/7b-context-extension-n32-hybrid-log_evo1-1M/7b-hybrid-log_evo1-1M_mp1/global_step12500/mp_rank_00_model_states.pt'

    # config_path = '/large_storage/hielab/gbrixi/checkpoints/evo2_interleaved/40b/global_step428000/40b_train_8K.yml'
    # new_checkpoint_path = '/scratch/hielab/gbrixi/evo2/vortex_interleaved/40b/'
    # checkpoint_path = '/large_storage/hielab/gbrixi/checkpoints/evo2_interleaved/40b/global_step428000/global_step428000/mp_rank_00_model_states.pt'
    # iteration = 428_000

    # checkpoint_path = '/large_storage/hielab/brianhie/checkpoints/7b-context-extension-n32-v3-hybrid-log_evo1-512K-cp-fix/7b-hybrid-log_evo1-512K-cp-fix_mp1/global_step12500/mp_rank_00_model_states.pt'
    # config_path='/scratch/hielab/gbrixi/evo2/vortex_interleaved/7b_524k/7b-hybrid-log_evo1-512K.yml'
    # new_checkpoint_path = '/scratch/hielab/gbrixi/evo2/vortex_interleaved/7b_524k/'
    # iteration = 12500

    # checkpoint_path = '/large_storage/hielab/brianhie/checkpoints/40b-train-n256-8K/40b_train_8K_zero1_mp1/global_step504000/mp_rank_00_model_states.pt'
    # config_path='/large_storage/hielab/brianhie/checkpoints/40b-train-n256-8K/40b_train_8K/global_step504000/configs/40b_train_8K.yml'
    # new_checkpoint_path = '/scratch/hielab/gbrixi/evo2/vortex_interleaved/40b/'
    # iteration = 504000

    ##converted on jeromeku/40b-bias-sync
    checkpoint_path = '/large_storage/hielab/gbrixi/checkpoints/evo2_interleaved/40b_1m_synced/mp1/global_step900/mp_rank_00_model_states.pt'
    config_path='/large_storage/hielab/gbrixi/checkpoints/evo2_interleaved/40b_1m_synced/global_step900/configs/40b_1M.yml'
    new_checkpoint_path = '/scratch/hielab/gbrixi/evo2/vortex_interleaved/40b_1m_synced/'
    iteration = 900

    def replace_hyphens_with_underscores(data: Any) -> Any:
        if isinstance(data, dict):
            return {key.replace('-', '_'): replace_hyphens_with_underscores(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [replace_hyphens_with_underscores(element) for element in data]
        else:
            return data
    # Load config
    with open(config_path, 'r') as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        yaml_config = replace_hyphens_with_underscores(yaml_config)
        yaml_config['model_parallel_size'] = 1
        yaml_config['load'] = None
        yaml_config['finetune'] = True

    os.environ["WORLD_SIZE"] = "1"
    os.environ["SLURM_NTASKS"] = "1"
    os.environ["SLURM_NTASKS_PER_NODE"] = "1"
    os.environ["GLOBAL_NUM_GPUS"] = "1"

    config = GlobalConfig(**yaml_config)

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    module_dict = state_dict['module']
    module_dict = organize_blocks(module_dict)

    # Process blocks
    new_state_dict = OrderedDict()
    block_dicts = {}
    current_block = {}
    current_block_idx = None
    block_dicts = {}
    current_block = {}
    current_block_idx = None
    
    for key, value in module_dict.items():
        if key.startswith('sequential.'):
            parts = key.split('.')
            block_idx = int(parts[1])
            
            if current_block_idx is not None and block_idx != current_block_idx:
                block_dicts[current_block_idx] = current_block
                current_block = {}
                
            current_block_idx = block_idx
            new_key = '.'.join(parts[2:])
            current_block[new_key] = value

    if current_block:
        block_dicts[current_block_idx] = current_block

    # Convert each block
    new_state_dict = OrderedDict()
    layer_counter = 0
    # Debug print
    print(f"Found {len(block_dicts)} blocks")
    for block_idx in sorted(block_dicts.keys()):
        print(f"Block {block_idx} keys: {block_dicts[block_idx].keys()}")

    for block_idx, block_dict in sorted(block_dicts.items()):
        operator_type = detect_module_cls(block_dict.keys())
        print(f"\nProcessing Block {block_idx}: Detected {operator_type}")
        
        converted_dict = convert_module_state_dict(block_dict, operator_type, config)
        
        if operator_type == "embedding":
            print(f"Adding embedding layer")
            for k, v in converted_dict.items():
                new_state_dict[k] = v
                new_state_dict["unembed.weight"] = v
        elif operator_type == "norm":
            print(f"Adding norm layer")
            for k, v in converted_dict.items():
                new_state_dict[k] = v
        else:  # blocks
            print(f"Adding block {layer_counter}")
            for k, v in converted_dict.items():
                new_key = f"blocks.{layer_counter}.{k}"
                print(f"Converting {k} -> {new_key}")
                new_state_dict[new_key] = v
            layer_counter += 1

    print("\nFinal layer counter:", layer_counter)
    print("Final keys:", new_state_dict.keys())

    # Save the checkpoint
    os.makedirs(new_checkpoint_path, exist_ok=True)
    checkpoint_file = f"iter_{iteration}.pt"
    torch.save(new_state_dict, os.path.join(new_checkpoint_path, checkpoint_file))

if __name__ == "__main__":
    main()