import gc
import re

import torch
import os

from clearml import Task


def save_training_state_with_clearml(task, model, optimizer, aux_optimizer,
                                     scheduler, aux_scheduler, current_epoch: int,
                                     current_step: int, save_path: str):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'aux_optimizer': aux_optimizer.state_dict() if aux_optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'aux_scheduler': aux_scheduler.state_dict() if aux_scheduler else None,
        'epoch': current_epoch,
        'step': current_step,
        'clearml_task_id': task.id if task is not None else None,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state_dict, save_path)
    if task:
        print(f"[INFO] Saved training state (ClearML task ID: {task.id}) to {save_path}")

def load_training_state_with_clearml_from_file(local_path, model, optimizer, aux_optimizer,
                                     scheduler, aux_scheduler, device):

    print(f"[INFO] Loading checkpoint from {local_path} to CPU first...")

    state = torch.load(local_path, map_location='cpu')
    print("[INFO] Checkpoint loaded to CPU.")

    task_id = state.get("clearml_task_id")

    model.to(device)
    print(f"[INFO] Model moved to {device}.")

    torch.cuda.empty_cache()
    gc.collect()

    model.load_state_dict(state['model'])
    print("[INFO] Model state_dict loaded.")
    del state['model']
    torch.cuda.empty_cache()
    gc.collect()

    if optimizer and 'optimizer' in state and state['optimizer']:
        print("[INFO] Loading optimizer state_dict...")

        optimizer.load_state_dict(state['optimizer'])
        print("[INFO] Optimizer state_dict loaded.")
        del state['optimizer']
        torch.cuda.empty_cache()
        gc.collect()
        optimizer.zero_grad(set_to_none=True)

    if aux_optimizer and 'aux_optimizer' in state and state['aux_optimizer']:
        print("[INFO] Loading aux_optimizer state_dict...")
        aux_optimizer.load_state_dict(state['aux_optimizer'])
        print("[INFO] Aux_optimizer state_dict loaded.")
        del state['aux_optimizer']
        torch.cuda.empty_cache()
        gc.collect()
        aux_optimizer.zero_grad(set_to_none=True)


    if scheduler and 'scheduler' in state and state['scheduler']:
        print("[INFO] Loading scheduler state_dict...")
        scheduler.load_state_dict(state['scheduler'])
        del state['scheduler']
        print("[INFO] Scheduler state_dict loaded.")

    if aux_scheduler and 'aux_scheduler' in state and state['aux_scheduler']:
        print("[INFO] Loading aux_scheduler state_dict...")
        aux_scheduler.load_state_dict(state['aux_scheduler'])
        del state['aux_scheduler']
        print("[INFO] Aux_scheduler state_dict loaded.")

    start_epoch = state['epoch'] + 1
    start_step = state['step']

    if task_id:
        task = Task.get_task(task_id=task_id)
        print(f"[INFO] Loaded ClearML Task: {task.name} (id={task_id})")
    else:
        task = None

    print(f"[INFO] Successfully loaded training state from epoch {start_epoch - 1}")

    del state
    torch.cuda.empty_cache()
    gc.collect()

    return task, start_epoch, start_step
