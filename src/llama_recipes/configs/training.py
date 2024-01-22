# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="NousResearch/Llama-2-70b-hf"
    enable_fsdp: bool=True
    low_cpu_fsdp: bool=True
    run_validation: bool=True 
    val_steps: int=12 # Run validation every val_steps training steps
    batch_size_training: int=16
    batching_strategy: str="padding" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int=1
    num_epochs: int=2
    num_workers_dataloader: int=1
    lr: float=2e-5
    lr_scheduler: str="cosine"
    restarts: bool=False
    warump_factor: float=0.04 # Only matter when using cosine lr_scheduler. Multiplier of total training steps to spend warming up LR
    decay_factor: float=10.0 # Only matter when using cosine lr_scheduler. The learning rate decays to LR times this factor from the end of warmup until the end of training.
    weight_decay: float=0.1
    gamma: float= 0.85
    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_method: str = "logits"
    val_batch_size: int=4
    dataset = "byjus_dataset"
    peft_method: str = "None" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "/home/checkpoints/20k_wd_01_bs16_lr_2e-5"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    checkpoint_path: str = ""
    optimizer_checkpoint_path: str = "" # If not empty string, start training again from this checkpoint
    save_model: bool = False
    dist_checkpoint_root_folder: str="/home/checkpoints/20k_wd_01_bs16_lr_2e-5/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="ft-model" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = True # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    wandb_project: str = "llama-70b-full-ft_20k"
    wandb_run_name: str = "wd_01_bs16_lr_2e-5"
    logging_steps: int = 1
