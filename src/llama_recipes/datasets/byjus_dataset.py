from datasets import Dataset
from dataclasses import dataclass, field
from typing import List, Dict
import json
import os
import sys
from copy import deepcopy

local_rank = os.environ['LOCAL_RANK']

@dataclass
class Question:
    question: str
    choices: List[str]
    answer: int
    read_more_content: str = ""  # Default is an empty string
    reasoning: str = ""  # Default is an empty string
    level: str = ""  # Default is an empty string
    similar_questions: List[str] = field(default_factory=list)  # Default is None
    similarity: float = 0.0  # Default is 0.0

    def format_question(self, prompt_prefix='', prompt_suffix='', use_readmore=False, use_reasoning=False, include_answer=False, n_shot=0) -> List[Dict[str, str]]:
        # Format the question
        formatted_question = self.question
        
        if prompt_prefix == "" and n_shot > 0:
            similar_questions = [from_dict(q) for q in self.similar_questions]
            similar_questions = [q for q in similar_questions if q.reasoning != "" and q.similarity < 0.95][:n_shot][::-1]
            prompt_prefix = "\n----------\n".join(similar_q.format_question(prompt_suffix='',
                                                                            use_readmore=use_readmore, 
                                                                            use_reasoning=use_reasoning,
                                                                            include_answer=True)
                                              for similar_q in similar_questions)
            prompt_prefix += "\n----------"
        
        # Format the choices with (a), (b), (c), etc
        choice_prefixes = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']
        for idx, choice in enumerate(self.choices):
            formatted_question += f"\n{choice_prefixes[idx]} {choice}"

        # Add read_more and reasoning if they're not empty
        if self.read_more_content and use_readmore:
            formatted_question += f"\n\nRead More: {self.read_more_content}"
            
        if self.reasoning != "" and self.reasoning is not None and use_reasoning:
            formatted_question += f"\n\nReasoning: {self.reasoning}"
            
        if include_answer:
            formatted_question += f"\n\nAnswer: {choice_prefixes[self.answer]}"
        
        # Return the complete formatted question
        return_val = (prompt_prefix + "\n" + formatted_question + "\n" + prompt_suffix).strip()
        return return_val
    

def from_dict(input_dict):
    """
    Initialize an instance of Question from a dictionary.
    Only keys that match the Question attributes will be considered.
    """
    # Extract the keys that are fields in the data class
    valid_keys = set(Question.__dataclass_fields__.keys())
    valid_data = {k: v for k, v in input_dict.items() if k in valid_keys}
    
    # Convert to data class
    return Question(**valid_data)

def tokenize_batch(examples, tokenizer, reasoning=False, all_tokens=False):
    # Tokenize the content
    input_ids = []
    if type(examples['text']) == str:
        examples = [examples['text']]
    else:
        examples = examples['text']
    tokenizer_outs = tokenizer(examples, add_special_tokens=False, padding=False, return_attention_mask=False)
    if type(tokenizer_outs['input_ids'][0]) != list:
        input_ids = [tokenizer_outs['input_ids']]
    else:
        input_ids = tokenizer_outs['input_ids']
    input_ids = [[tokenizer.bos_token_id] + t + [tokenizer.eos_token_id] for t in input_ids]
    attention_mask = [[1] * (len(chunk)) for chunk in input_ids]
    if all_tokens:
        labels = deepcopy(input_ids)
    else:
        labels = [[-100] * (len(t) - 3) + [t[-3]] + [-100] * 2 for t in input_ids]
    if len(input_ids) == 1:
        input_ids = input_ids[0]
        labels = labels[0]
        attention_mask = attention_mask[0]
    return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}

def get_byjus_dataset(dataset_config, tokenizer, split):
    def tokenize_batch_train(examples):
        return tokenize_batch(examples, tokenizer, all_tokens=dataset_config.all_tokens)
    def tokenize_batch_val(examples):
        return tokenize_batch(examples, tokenizer, all_tokens=dataset_config.all_tokens)
    if split == 'train':
        with open(dataset_config.train_data_path, "r") as f:
            train_qs = json.load(f)
        train_qs = [from_dict(q) for q in train_qs]
        dataset_dict = {'text': [q.format_question(include_answer=True, use_reasoning=dataset_config.use_reasoning_in_train) for q in train_qs if q.answer in [0, 1, 2, 3]]}
        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.map(tokenize_batch_train, batched=True, load_from_cache_file=True, remove_columns=['text'])
        return dataset
    else:
        datasets = []
        for eval_path in dataset_config.eval_data_paths:
            with open(eval_path, "r") as f:
                eval_qs = json.load(f)
            eval_qs = [from_dict(q) for q in eval_qs]
            dataset_dict = {'text': [q.format_question(prompt_prefix='', include_answer=True, n_shot=dataset_config.n_shot) for q in eval_qs]}
            dataset = Dataset.from_dict(dataset_dict)
            dataset = dataset.map(tokenize_batch_val, batched=True, load_from_cache_file=True, remove_columns=['text'])
            datasets.append(dataset)
            if dataset_config.run_cot_eval:
                dataset_dict_cot = {'text': [q.format_question(prompt_prefix='', prompt_suffix='\nReasoning:', use_reasoning=True, include_answer=False, n_shot=dataset_config.n_shot) for q in eval_qs]}
                dataset_cot = Dataset.from_dict(dataset_dict_cot)
                dataset_cot = dataset_cot.map(tokenize_batch_val, batched=True, load_from_cache_file=True, remove_columns=['text'])
                dataset_cot = dataset_cot.remove_columns(['labels'])
                dataset_cot = dataset_cot.add_column('labels', labels)
                labels = deepcopy(dataset['labels'])
                datasets.append(dataset_cot)
        if dataset_config.run_cot_eval:
            return {"PYQ_2022": datasets[0], "PYQ_2022_CoT": datasets[1], "PYQ_2023": datasets[2], "PYQ_2023_CoT": datasets[3]}
        else:
            return {"PYQ_2022": datasets[0], "PYQ_2023": datasets[1]}