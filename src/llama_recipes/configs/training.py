# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="NousResearch/Llama-2-7b-hf"
    enable_fsdp: bool=True
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=8
    batching_strategy: str="packing" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int=1
    num_epochs: int=3
    num_workers_dataloader: int=1
    lr: float=5e-3
    lr_scheduler: str="cosine"
    weight_decay: float=0.0
    gamma: float= 0.85
    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "custom_dataset"
    peft_method: str = "None" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "/home/checkpoints/wiki-india-full-ft_1e-3"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="/home/checkpoints/wiki-india-full-ft_1e-3/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="ft-model" # will be used if using FSDP
    save_optimizer: bool=True # will be used if using FSDP
    use_fast_kernels: bool = True # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    wandb_project: str = "llama-7b-full-ft"
    wandb_run_name: str = "wiki-india-lr_1e-3"
    logging_steps: int = 5
