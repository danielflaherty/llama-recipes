import os
import math
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, peak_lr: float, decay_factor: float, num_cycles: float
):
    min_lr = peak_lr / decay_factor  # Minimum learning rate
    max_lr = peak_lr  # Maximum learning rate
    scale = max_lr - min_lr

    # Linear warmup
    if current_step < num_warmup_steps:
        progress = float(current_step) / float(max(1, num_warmup_steps))
        return progress + ((1.0 - progress) * min_lr) / max_lr

    # Cosine decay
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_decay = max(min_lr, min_lr + 0.5 * scale * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return cosine_decay / max_lr

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, peak_lr: float, decay_factor: float, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        peak_lr=peak_lr,
        decay_factor=decay_factor,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)