import os
import sys
import time

import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.model_utils import load_model

from dataclasses import dataclass
from typing import List, Dict
import json

# Set the seeds for reproducibilitys
seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

@dataclass
class Question:
    question: str
    choices: List[str]
    answer: int
    read_more_content: str = ""  # Default is an empty string
    reasoning: str = ""  # Default is an empty string
    level: str = ""  # Default is an empty string

    def format_question(self, prompt_prefix='', prompt_suffix='', use_readmore=False, use_reasoning=False) -> List[Dict[str, str]]:
        # Format the question
        formatted_question = self.question
        
        # Format the choices with (a), (b), (c), etc
        choice_prefixes = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']
        for idx, choice in enumerate(self.choices):
            formatted_question += f"\n{choice_prefixes[idx]} {choice}"

        # Add read_more and reasoning if they're not empty
        if self.read_more_content and use_readmore:
            formatted_question += f"\n\nRead More: {self.read_more_content}"
            
        if self.reasoning and use_reasoning:
            formatted_question += f"\nExplain your reasoning, then output your answer at the end:\n\nReasoning: {self.reasoning}"
        
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

with open("/home/danielflaherty/llama-recipes/eval_data/PYQ_2022.json", "r") as f:
    eval_qs = json.load(f)
with open("/home/danielflaherty/llama-recipes/eval_data/prompts/5_shot_CoT.txt", 'r') as f:
    prompt_prefix = f.read()
eval_qs = [from_dict(q) for q in eval_qs]
ans_lst = ["(a)", "(b)", "(c)", "(d)"]
inputs = [q.format_question(prompt_prefix=prompt_prefix, prompt_suffix="Explain your reasoning, then output your answer at the end:\n\nReasoning:") for q in eval_qs]
answers = [ans_lst[q.answer] for q in eval_qs]

model_paths = {"epoch_1": ["/home/checkpoints/reasoning_cot_wiki_all_epoch_1_sft_byjus_all/FSDP/model/ft-model-NousResearch/Llama-2-7b-hf-epoch_0/NousResearch/Llama-2-7b-hf-0.pt",
                           "/home/checkpoints/reasoning_cot_wiki_all_epoch_1_sft_byjus_all/FSDP/model/ft-model-NousResearch/Llama-2-7b-hf-epoch_1/NousResearch/Llama-2-7b-hf-1.pt",
                           "/home/checkpoints/reasoning_cot_wiki_all_epoch_1_sft_byjus_all/FSDP/model/ft-model-NousResearch/Llama-2-7b-hf-epoch_2/NousResearch/Llama-2-7b-hf-2.pt"],
               "epoch_2": ["/home/checkpoints/reasoning_cot_wiki_all_epoch_2_sft_byjus_all/FSDP/model/ft-model-NousResearch/Llama-2-7b-hf-epoch_0/NousResearch/Llama-2-7b-hf-0.pt",
                           "/home/checkpoints/reasoning_cot_wiki_all_epoch_2_sft_byjus_all/FSDP/model/ft-model-NousResearch/Llama-2-7b-hf-epoch_1/NousResearch/Llama-2-7b-hf-1.pt",
                           "/home/checkpoints/reasoning_cot_wiki_all_epoch_2_sft_byjus_all/FSDP/model/ft-model-NousResearch/Llama-2-7b-hf-epoch_2/NousResearch/Llama-2-7b-hf-2.pt"],
               "epoch_3": ["/home/checkpoints/reasoning_cot_wiki_all_epoch_3_sft_byjus_all/FSDP/model/ft-model-NousResearch/Llama-2-7b-hf-epoch_0/NousResearch/Llama-2-7b-hf-0.pt",
                           "/home/checkpoints/reasoning_cot_wiki_all_epoch_3_sft_byjus_all/FSDP/model/ft-model-NousResearch/Llama-2-7b-hf-epoch_1/NousResearch/Llama-2-7b-hf-1.pt",
                           "/home/checkpoints/reasoning_cot_wiki_all_epoch_3_sft_byjus_all/FSDP/model/ft-model-NousResearch/Llama-2-7b-hf-epoch_2/NousResearch/Llama-2-7b-hf-2.pt"]}

epoch_results = {"epoch_1": [], "epoch_2": [], "epoch_3": []}
for epoch, path_list in model_paths.items():
    for model_path in path_list:
        model = load_model("NousResearch/Llama-2-7b-hf", False)
        model_checkpoint = torch.load(model_path)
        # integrate into loaded model
        model.load_state_dict(model_checkpoint)
        model.eval()
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
        num_correct = 0
        for i, q in enumerate(inputs):
            input_ids = tokenizer.encode(q, return_tensors="pt")
            inputs_ids = input_ids.to("cuda")
            with torch.no_grad():
                outputs = model.generate(input_ids, do_sample=False, max_new_tokens=250, return_dict_in_generate=False)
                output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if answers[i] in output_str[-75:]:
                    num_correct += 1
        epoch_results[epoch].append(num_correct/len(inputs))
        torch.cuda.empty_cache()
            
with open('reasoning_cot_eval_results.json', 'w') as f:
    json.dump(epoch_results, f)