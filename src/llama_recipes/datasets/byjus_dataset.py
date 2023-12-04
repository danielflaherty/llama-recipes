from datasets import Dataset
from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class Question:
    question: str
    choices: List[str]
    answer: int
    read_more_content: str = ""  # Default is an empty string
    reasoning: str = ""  # Default is an empty string
    level: str = ""  # Default is an empty string

    def format_question(self, prompt_prefix='', prompt_suffix='', use_readmore=False) -> List[Dict[str, str]]:
        # Format the question
        formatted_question = self.question
        
        # Format the choices with (a), (b), (c), etc
        choice_prefixes = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']
        for idx, choice in enumerate(self.choices):
            formatted_question += f"\n{choice_prefixes[idx]} {choice}"

        # Add read_more and reasoning if they're not empty
        if self.read_more_content and use_readmore:
            formatted_question += f"\n\nRead More: {self.read_more_content}"
        
        # Return the complete formatted question
        return (prompt_prefix + "\n" + formatted_question + "\n" + prompt_suffix).strip()
    

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

def tokenize_batch(examples, tokenizer, train=True):
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
    if train:
        labels = [[-100] * (len(t) - 4) + t[-4:] for t in input_ids]
    else:
        labels = [[-100] * (len(t) - 3) + [t[-3]] + [-100] * 2 for t in input_ids]
    if len(input_ids) == 1:
        input_ids = input_ids[0]
        labels = labels[0]
        attention_mask = attention_mask[0]
    return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}

def get_byjus_dataset(dataset_config, tokenizer, split):
    def tokenize_batch_train(examples):
        return tokenize_batch(examples, tokenizer, train=True)
    def tokenize_batch_val(examples):
        return tokenize_batch(examples, tokenizer, train=False)
    if split == 'train':
        with open(dataset_config.train_data_path, "r") as f:
            train_qs = json.load(f)
        train_qs = [from_dict(q) for q in train_qs]
        ans_lst = ["(a)", "(b)", "(c)", "(d)"]
        dataset_dict = {'text': [q.format_question(prompt_suffix=f"Answer: {ans_lst[q.answer]}") for q in train_qs]}
        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.map(tokenize_batch_train, batched=True, load_from_cache_file=True, remove_columns=['text'])
    else:
        with open(dataset_config.eval_data_path, "r") as f:
            eval_qs = json.load(f)
        with open(dataset_config.eval_data_prefix_path, 'r') as f:
            prompt_prefix = f.read()
        eval_qs = [from_dict(q) for q in eval_qs]
        ans_lst = ["(a)", "(b)", "(c)", "(d)"]
        dataset_dict = {'text': [q.format_question(prompt_prefix=prompt_prefix, prompt_suffix=f"Answer: {ans_lst[q.answer]}") for q in eval_qs]}
        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.map(tokenize_batch_val, batched=True, load_from_cache_file=True, remove_columns=['text'])

    return dataset