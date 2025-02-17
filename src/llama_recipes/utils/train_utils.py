# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
import wandb
from dataclasses import dataclass, asdict
import math
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import datetime
import math


from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)

def wandb_watch(train_config, fsdp_config=None, rank=None):
    run = wandb.init(project=train_config.wandb_project,
                    name=train_config.wandb_run_name,
                    config=asdict(train_config) if fsdp_config is None else {**asdict(train_config), **asdict(fsdp_config)})
    return run

def get_gradient_norm(model):
    """
    Calculate the gradient norm of an FSDP model.

    Args:
    - model (torch.nn.Module): The FSDP model.

    Returns:
    - float: The total norm of the gradients.
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm

def run_both_evals(eval_dataloaders, model, train_config, tokenizer, local_rank, rank, step, epoch):
    eval_metrics = {}
    for dataset_name, eval_dataloader in eval_dataloaders.items():  
        if "CoT" in dataset_name:
            acc, cot_preds = evaluation_CoT(model, train_config, eval_dataloader, local_rank, tokenizer)
        else:
            eval_ppl, eval_epoch_loss, acc, other_tokens_loss, probs_ratio = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer)
        eval_metrics.update({f'eval/{dataset_name}_epoch_loss': eval_epoch_loss, f"eval/{dataset_name}_other_tokens_loss": other_tokens_loss,
                             f"eval/{dataset_name}_model_wrong_probs_ratio": probs_ratio, f'eval/{dataset_name}_perplexity': eval_ppl})
        eval_metrics.update({f'eval/{dataset_name}_accuracy': acc})
        if (train_config.enable_fsdp and rank == 0) or not train_config.enable_fsdp:
            eval_metrics.update({'eval/global_step': step, 'eval/epoch': epoch})
            if "CoT" in dataset_name:
                table = wandb.Table(columns=["input", "output"])
                for input_str, output_str in cot_preds.items():
                    table.add_data(input_str, output_str)
                wandb.log({f"Epoch_{epoch}_Preds": table}, commit=False)
            wandb.log(eval_metrics)
    return eval_metrics

def train(model, train_dataloader, eval_dataloaders, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloaders: The dataloaders containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    epoch_times = []
    checkpoint_times = []
    best_val_loss = float("inf")
    
    # For Calculating Accuracy:
    a_tok = tokenizer.encode("(a)")[-2]
    b_tok = tokenizer.encode("(b)")[-2]
    c_tok = tokenizer.encode("(c)")[-2]
    d_tok = tokenizer.encode("(d)")[-2]
    ans_toks = [a_tok, b_tok, c_tok, d_tok]
            
    # wandb setuo
    n_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    if train_config.enable_fsdp and rank == 0:
        wandb_watch(train_config, fsdp_config, rank)
        
    # Run evaluation before training 
    if train_config.run_validation:
        run_both_evals(eval_dataloaders, model, train_config, tokenizer, local_rank, rank, 0, 0.0)
    # Training loop
    total_steps = 0
    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = torch.tensor(0.0).cuda()
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader):
                for key in batch.keys():
                    if train_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        batch[key] = batch[key].to('cuda:0')
                with autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
                # Calculate train accuracy:
                ans_2_idx = {"a": 0, "b": 1, "c": 2, "d": 3}
                labels = []
                num_correct = 0
                other_tokens_loss = 0.0
                batch_size = batch['labels'].shape[0]
                for i in range(batch_size):
                    labels_list = batch['labels'][i].tolist()
                    label_idx = find_matching_index(labels_list, ans_toks)
                    label = labels_list[label_idx]
                    labels.append(label)
                    correct_ans = tokenizer.decode(label)
                    correct_ans = ans_2_idx[correct_ans]
                    logprobs = torch.nn.functional.log_softmax(outputs.logits[i, label_idx - 1, :], dim=0)
                    ans_logits = [logprobs[a_tok].item(), logprobs[b_tok].item(), logprobs[c_tok].item(), logprobs[d_tok].item()]
                    other_tokens_loss -= sum(ans_logits) + ans_logits[correct_ans]
                    max_logprob = max(ans_logits)
                    model_ans = ans_logits.index(max_logprob)
                    ans_logits.sort(reverse=True)
                    if model_ans == correct_ans:
                        num_correct += 1                    
                if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
                    num_correct = torch.tensor(num_correct).cuda()
                    other_tokens_loss = torch.tensor(other_tokens_loss).cuda()
                    batch_size = torch.tensor(batch_size).cuda()
                    dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
                    dist.all_reduce(batch_size, op=dist.ReduceOp.SUM)
                    dist.all_reduce(other_tokens_loss, op=dist.ReduceOp.SUM)
                acc = num_correct.item() / batch_size.item()
                other_tokens_loss = other_tokens_loss.item() / batch_size.item()
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                             scaler.unscale_(optimizer)
                             if train_config.enable_fsdp:
                                 model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                             else:
                                 torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                        scaler.step(optimizer)
                        scaler.update()
                        pbar.update(1)
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                             if train_config.enable_fsdp:
                                 model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                             else:
                                 torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                        optimizer.step()
                        
                epoch_step = ((step + 1) / n_steps_per_epoch) + epoch
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    total_steps += 1
                if train_config.lr_scheduler == "cosine" and (step + 1) % gradient_accumulation_steps == 0:
                    lr_scheduler.step()
                    
                # Log to wandb
                pbar.update(1)
                grad_norm = torch.tensor(get_gradient_norm(model)).cuda()
                if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
                    dist.all_reduce(grad_norm, op=dist.ReduceOp.SUM)
                grad_norm = grad_norm.item() ** 0.5
                
                # Now that we've calculated grad norm, we can zero the gradients
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.zero_grad()
                    
                loss = loss.detach().float()
                if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / torch.cuda.device_count()
                if rank == 0 and (step + 1) % train_config.logging_steps == 0:
                    metrics = {"train/train_loss:": loss, 
                               "train/train_accuracy": acc,
                               "train/train_other_tokens_loss": other_tokens_loss,
                               "train/train_perplexity": torch.exp(loss),
                                "train/epoch": epoch_step,
                                "train/global_step": total_steps,
                                "train/lr": optimizer.param_groups[0]['lr'],
                                "train/grad_norm": grad_norm,
                              }
                    wandb.log(metrics)
                    
                # Run validation
                if train_config.run_validation and total_steps % train_config.val_steps == 0:
                    eval_metrics = run_both_evals(eval_dataloaders, model, train_config, tokenizer, local_rank, rank, total_steps, epoch + ((step + 1) / n_steps_per_epoch))
                    
                pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
            pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        if train_config.enable_fsdp:
            if rank==0:
                print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        else:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")

        # Update the learning rate as needed
        if train_config.lr_scheduler == "step":
            lr_scheduler.step()

        if train_config.save_model:
            eval_epoch_loss = float("inf")
            if train_config.run_validation:
                eval_metrics = run_both_evals(eval_dataloaders, model, train_config, tokenizer, local_rank, rank, total_steps, epoch + 1)
                eval_epoch_loss = eval_metrics['eval/PYQ_2022_epoch_loss']
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                checkpoint_start_time = time.perf_counter()
                if train_config.enable_fsdp:
                    dist.barrier()
                if train_config.use_peft:
                    if train_config.enable_fsdp:
                        if rank==0:
                            print(f"we are about to save the PEFT modules")
                    else:
                        print(f"we are about to save the PEFT modules")
                    model.save_pretrained(train_config.output_dir)
                    if train_config.enable_fsdp:
                        if rank==0:
                            print(f"PEFT modules are saved in {train_config.output_dir} directory")
                    else:
                        print(f"PEFT modules are saved in {train_config.output_dir} directory")

                else:
                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                        save_model_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                        print("=====================================================")

                        save_model_and_optimizer_sharded(model, rank, train_config, epoch)
                        if train_config.save_optimizer:
                            save_model_and_optimizer_sharded(model, rank, train_config, epoch, optim=optimizer)
                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                            print("=====================================================")

                    if not train_config.use_peft and  train_config.save_optimizer:
                        save_optimizer_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                        print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                        print("=====================================================")
                if train_config.enable_fsdp:
                    dist.barrier()
                checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                checkpoint_times.append(checkpoint_end_time)

        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={float(train_perplexity):.4f}, train_epoch_loss={float(train_epoch_loss):.4f}, epoch time {float(epoch_end_time)}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={float(train_perplexity):.4f}, train_epoch_loss={float(train_epoch_loss):.4f}, epoch time {float(epoch_end_time)}s")

    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)
        
    if train_config.enable_fsdp and rank == 0:
        wandb.finish()
        
def find_matching_index(list1, list2):
    """
    Function to find the first index in list1 (working from the end back to the left)
    where one of its numbers equals a number in list2.

    :param list1: The first list of numbers.
    :param list2: The second list of numbers.
    :return: The index in list1 where a match is found, or None if no match is found.
    """
    for i in range(len(list1) - 1, -1, -1):
        if list1[i] in list2:
            return i
    return None

def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    num_correct = 0
    total_pred = 0
    eval_loss = 0.0  # Initialize evaluation loss
    eval_other_tokens_loss = 0.0
    probs_ratio = 0.0

    # For calculating Accuracy:
    a_tok = tokenizer.encode("(a)")[-2]
    b_tok = tokenizer.encode("(b)")[-2]
    c_tok = tokenizer.encode("(c)")[-2]
    d_tok = tokenizer.encode("(d)")[-2]
    ans_toks = [a_tok, b_tok, c_tok, d_tok]

    
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
            ans_2_idx = {"a": 0, "b": 1, "c": 2, "d": 3}
            labels = []
            for i in range(batch['labels'].shape[0]):
                labels_list = batch['labels'][i].tolist()
                label_idx = find_matching_index(labels_list, ans_toks)
                label = labels_list[label_idx]
                labels.append(label)
                correct_ans = tokenizer.decode(label)
                correct_ans = ans_2_idx[correct_ans]
                logprobs = torch.nn.functional.log_softmax(outputs.logits[i, label_idx - 1, :], dim=0)
                ans_logits = [logprobs[a_tok].item(), logprobs[b_tok].item(), logprobs[c_tok].item(), logprobs[d_tok].item()]
                eval_other_tokens_loss -= sum(ans_logits) + ans_logits[correct_ans]
                correct_ans_logprob = ans_logits[correct_ans]
                max_logprob = max(ans_logits)
                model_ans = ans_logits.index(max_logprob)
                ans_logits.sort(reverse=True)
                if model_ans == correct_ans:
                    num_correct += 1 
                    probs_ratio += 1.0             
                else:
                    probs_ratio += math.exp(correct_ans_logprob - max_logprob)
            total_pred += len(batch['input_ids'])
    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Compute accuracy
    num_correct = torch.tensor(num_correct).cuda()
    total_pred = torch.tensor(total_pred).cuda()
    eval_other_tokens_loss = torch.tensor(eval_other_tokens_loss).cuda()
    probs_ratio = torch.tensor(probs_ratio).cuda()
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_pred, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_other_tokens_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(probs_ratio, op=dist.ReduceOp.SUM)
    acc = num_correct.item() / total_pred.item()
    eval_other_tokens_loss = eval_other_tokens_loss.item() / total_pred.item()
    if total_pred.item() - num_correct.item() > 0:
        probs_ratio = probs_ratio.item() / (total_pred.item() - num_correct.item())
    else:
        probs_ratio = 0.0

    return eval_ppl, eval_epoch_loss, acc, eval_other_tokens_loss, probs_ratio

def evaluation_CoT(model, train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    model.eval()
    num_correct = 0
    cot_preds = {}
    total_pred = 0
    
    # For calculating Accuracy:
    a_tok = tokenizer.encode("(a)")[-2]
    b_tok = tokenizer.encode("(b)")[-2]
    c_tok = tokenizer.encode("(c)")[-2]
    d_tok = tokenizer.encode("(d)")[-2]

    
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')
            labels = []
            for i in range(batch['labels'].shape[0]):
                label_idx = (batch['labels'][i] != -100).nonzero()[0].item()
                label = batch['labels'][i][label_idx].item()
                labels.append(label)
            with torch.no_grad():
                # Summon Model Prams w/ dummy forward
                dummy_in = torch.ones_like(batch['input_ids']).to(local_rank)
                dummy_mask = torch.ones_like(batch['attention_mask']).to(local_rank)
                model(input_ids=dummy_in, attention_mask=dummy_mask)
                outputs = model.generate(batch['input_ids'], do_sample=False, return_dict_in_generate=False, max_new_tokens=250)
                input_strs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                output_strs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                output_strs = [output_strs[i].replace(input_strs[i], '') for i in range(len(input_strs))]
                cot_preds.update({input_strs[i]: output_strs[i] for i in range(len(input_strs))})
                ans_choices = {a_tok: "(a)", b_tok: "(b)", c_tok: "(c)", d_tok: "(d)"}
                for i in range(len(labels)):
                    correct_ans = ans_choices[label]
                    output_ans = output_strs[i].split("Answer:")[-1]
                    if correct_ans in output_ans:
                        num_correct += 1
            total_pred += len(batch['input_ids'])
    
    # Compute accuracy
    num_correct = torch.tensor(num_correct).cuda()
    total_pred = torch.tensor(total_pred).cuda()
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_pred, op=dist.ReduceOp.SUM)
    acc = num_correct.item() / total_pred.item()
    
    all_cot_preds = [None for _ in range(dist.get_world_size())]
    combined_cot_preds = {}
    dist.all_gather_object(all_cot_preds, cot_preds)
    for d in all_cot_preds:
        combined_cot_preds.update(d)
    cot_preds = combined_cot_preds

    return acc, cot_preds

def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=2800))
    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["NCCL_BLOCKING_WAIT"] = str(1)


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["NCCL_BLOCKING_WAIT"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")
