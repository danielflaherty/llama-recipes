# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from typing import List
from dataclasses import dataclass, field

    
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
    
@dataclass
class byjus_dataset:
    dataset: str = "byjus_dataset"
    file: str = "/home/danielflaherty/llama-recipes/src/llama_recipes/datasets/byjus_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    train_data_path: str = '/home/upsc-gpt-data/train_data/all_qs_filtered_plus_edugorilla.json'
    train_data_packing: bool = False
    eval_data_paths: List[str] = field(default_factory=list)
    run_cot_eval: bool = False
    n_shot: int = 5
    use_reasoning_in_train: bool = False
    all_tokens: bool = False
    def __post_init__(self):
        self.eval_data_paths = ["/home/danielflaherty/llama-recipes/eval_data/PYQ_2022_Similar_Qs.json", "/home/danielflaherty/llama-recipes/eval_data/PYQ_2023_Similar_Qs.json"]