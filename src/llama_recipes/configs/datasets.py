# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    
@dataclass
class wiki_india_dataset:
    dataset: str = "wiki_india_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    train_data_path: str = "/home/danielflaherty/upsc-gpt-data/train_data/wiki_india.txt"
    train_data_packing: bool = True
    eval_data_path: str = "/home/danielflaherty/upsc-gpt-data/eval_data/PYQ_2022.json"
    eval_data_prefix_path: str = "/home/danielflaherty/upsc-gpt-data/eval_data/prompts/5_shot.txt"