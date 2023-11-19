import copy
import os
import datasets
from datasets import Features, Value, Dataset
import itertools
from dataclasses import dataclass, field
from typing import List, Dict
import random
import json
import openai
from tenacity import retry, wait_exponential, stop_after_attempt

@dataclass
class Question:
    question_id: int
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

# Parse model output. If not found, then use GPT-4 to extract answer choice.
def get_majority_vote(question: str, model_outs: List[str], choices: List[str] = ['a', 'b', 'c', 'd', '<No Clear Answer>'], gpt_backup: bool = True):
    votes = {}
    for choice in choices:
        votes[choice] = 0
    for model_out in model_outs:
        # Try to find answer choice in last 100 characters of model output
        choice_found = False
        for choice in choices:
            # Quetion formats choices in parenthesis to make it easier to find. So search for ({choice}) rather than just {choice}.
            if f"({choice})" in model_out.lower():
                choice_found = True
                votes[choice] += 1
                break
        # If answer choice not found, then use GPT-4 to extract answer choice
        if not choice_found:
            if gpt_backup:
                ans, _ = get_majority_vote_gpt(question, [model_out], choices)
                votes[ans] += 1
            else:
                votes['<No Clear Answer>'] += 1
    return max(votes, key=votes.get), votes

def get_majority_vote_gpt(question: str, model_outs: List[str], choices=['a', 'b', 'c', 'd', '<No Clear Answer>']):
    votes = {}
    for choice in choices:
        votes[choice] = 0
    for model_out in model_outs:
        # Get GPT-4 to extract the answer
        func_description = "Function that OpenAI Model will use to input the extracted answer choice."
        parameters_dict = {'type': 'object', 
                           'properties': 
                                {'answer_letter': {'type': "string", 'enum': choices, 'description': 'Answer choice to be extracted from the given answer explanation. If no clear answer choice is found to the given question, then output <No Clear Answer>.'}
                                 }, 
                            'required': ['answer_letter'],
                            }
        function_dict = {'name': 'input_answer_letter', 'description': func_description, 'parameters': parameters_dict}
        input_str = f'Question:\n {question}\n\nAnswer Explanation:\n {model_out}'
        sys_prompt = "Your job is to take a look at a multiple-choice question, and a proposed answer explanation, and determine which of 'a', 'b', 'c', or 'd' that the question answerer chose. It is possible the explanation is non-sensical with regard to the question, in which case you should output <No Clear Answer>. Do NOT try to answer the question yourself."
        messages = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': input_str}]
        response = call_oai(messages, function_dict)
        function_args = json.loads(response['choices'][0]['message']["function_call"]["arguments"])
        answer_letter = function_args['answer_letter']
        for choice in choices:
            if choice.lower() == answer_letter.lower():
                votes[choice] += 1
                break
    return max(votes, key=votes.get), votes

@retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(5))
def call_oai(messages, function_dict):
    openai.api_key = keys[0]
    keys.append(keys.pop(0))
    response = openai.ChatCompletion.create(model='gpt-4', messages=messages, function_call={"name": 'input_answer_letter'}, functions=[function_dict])
    return response

def article_generator():
    file_path = "/home/danielflaherty/upsc-gpt-data/train_data/wiki_india.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        current_article_content = ""
        for line in file:
            if line == "\n" and current_article_content:
                # Yield the current article and reset for the next
                yield {'article': current_article_content.strip()}
                current_article_content = ""
            else:
                current_article_content += line

        # Yield the last article if there's no trailing newline
        if current_article_content:
            yield {'article': current_article_content.strip()}

def tokenize_and_chunk(examples, tokenizer):
    # Create chunks of 4096 tokens
    chunks = []
    for example in examples:
        # Tokenize the content
        tokens = tokenizer.encode(example['article'])
        chunks += [tokens[i:i+4096] for i in range(0, len(tokens), 4096)]
    return {'input_ids': chunks, 'labels': chunks, 'attention_mask': [[1] * len(chunk) for chunk in chunks]}

def tokenize_normal(examples, tokenizer):
    # Tokenize the content
    input_ids, labels = [], []
    examples = [example for example in examples['article']]
    tokenizer_outs = tokenizer(examples, add_special_tokens=False)
    input_ids = [[tokenizer.bos_token_id] + t + [tokenizer.eos_token_id] for t in tokenizer_outs['input_ids']]
    labels = [t[:-1] + [-100] for t in input_ids]
    return {'input_ids': input_ids, 'labels': labels}

def get_wiki_india_dataset(dataset_config, tokenizer, split):
    if split == 'train':
        features = Features({'article': Value('string')})
        dataset = Dataset.from_generator(article_generator, features)
        if True:
            dataset = dataset.map(lambda examples: tokenize_normal(examples, tokenizer), batched=True)
        else:
            dataset = dataset.map(lambda examples: tokenize_article(examples, tokenizer), batched=True)
    else:
        with open("/home/danielflaherty/upsc-gpt-data/eval_data/PYQ_2022.json", "r") as f:
            dataset = json.load(f)
        with open("/home/danielflaherty/upsc-gpt-data/eval_data/prompts/5_shot.txt", 'r') as f:
            prompt_prefix = f.read()
        dataset = [from_dict(q) for q in dataset]
        dataset = [{'question': q.format_question(prompt_prefix=prompt_prefix, prompt_suffix="Answer:"), 'answer': q.answer} for q in dataset]
        
        dataset_dict = {'text': [q['question'] for q in dataset], 'answer': [q['answer'] for q in dataset]}
        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.map(lambda sample: {'input_ids': tokenizer.encode(sample["text"])},
                                                batched=True, remove_columns=list(dataset.features))

    return dataset
