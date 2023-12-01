# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="NousResearch/Llama-2-7b-hf"
    enable_fsdp: bool=True
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    val_steps: int=100 # Run validation every val_steps training steps
    batch_size_training: int=8
    batching_strategy: str="packing" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int=1
    num_epochs: int=3
    num_workers_dataloader: int=1
    lr: float=1e-4
    lr_scheduler: str="cosine"
    warump_factor: float=0.02 # Only matter when using cosine lr_scheduler. Multiplier of total training steps to spend warming up LR
    decay_factor: float=30.0 # Only matter when using cosine lr_scheduler. The learning rate decays to LR times this factor from the end of warmup until the end of training.
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
    output_dir: str = "/home/checkpoints/fixed_wiki-india-full-ft_1e-4_beta2_0.95_warmup_0.02"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    checkpoint_path: str = "" # If not empty string, start training again from the checkpoint
    save_model: bool = True
    dist_checkpoint_root_folder: str="/home/checkpoints/fixed_wiki-india-full-ft_1e-4_beta2_0.95_warmup_0.02/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="ft-model" # will be used if using FSDP
    save_optimizer: bool=True # will be used if using FSDP
    use_fast_kernels: bool = True # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    wandb_project: str = "llama-7b-full-ft"
    wandb_run_name: str = "fixed_wiki-india-full-ft_1e-4_beta2_0.95_warmup_0.02"
    logging_steps: int = 5
