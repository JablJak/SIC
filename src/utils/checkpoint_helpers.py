import gc

import torch
import os

from clearml import Task


def save_training_state_with_clearml(task, model, optimizer, aux_optimizer,
                                     scheduler, aux_scheduler, scaler,
                                     current_epoch: int, save_path: str):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'aux_optimizer': aux_optimizer.state_dict() if aux_optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'aux_scheduler': aux_scheduler.state_dict() if aux_scheduler else None,
        'scaler': scaler.state_dict(),
        'epoch': current_epoch,
        'clearml_task_id': task.id if task is not None else None,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state_dict, save_path)
    if task:
        print(f"[INFO] Saved training state (ClearML task ID: {task.id}) to {save_path}")

def load_training_state_with_clearml_from_file(local_path, model, optimizer, aux_optimizer,
                                     scheduler, aux_scheduler, scaler, device):

    print(f"[INFO] Loading checkpoint from {local_path} to CPU first...")
    # Załaduj stan na CPU, aby uniknąć skoku zużycia pamięci GPU
    state = torch.load(local_path, map_location='cpu')
    print("[INFO] Checkpoint loaded to CPU.")

    task_id = state.get("clearml_task_id")
    if not task_id:
        raise ValueError("'clearml_task_id' not found in checkpoint")

    # Upewnij się, że model jest już na właściwym urządzeniu PRZED ładowaniem stanu
    model.to(device)
    print(f"[INFO] Model moved to {device}.")

    # Wyczyść cache GPU przed ładowaniem stanu modelu
    torch.cuda.empty_cache()
    gc.collect() # Dodatkowo wymuś odśmiecanie pamięci CPU

    print("[INFO] Loading model state_dict...")
    # with torch.no_grad(): # Nie jest konieczne dla load_state_dict
    model.load_state_dict(state['model'])
    print("[INFO] Model state_dict loaded.")
    del state['model'] # Usuń stan modelu z CPU, aby zwolnić RAM
    torch.cuda.empty_cache()
    gc.collect()

    # Przenieś stany optymalizatorów na GPU *po* załadowaniu ich do obiektów optymizatora
    # Optymalizatory same powinny zarządzać przenoszeniem swoich stanów na odpowiednie urządzenie
    # podczas load_state_dict, jeśli ich parametry (parametry modelu) są już na GPU.
    print("[INFO] Loading optimizer state_dict...")
    optimizer.load_state_dict(state['optimizer'])
    print("[INFO] Optimizer state_dict loaded.")
    del state['optimizer'] # Zwolnij RAM
    torch.cuda.empty_cache()
    gc.collect()

    if scheduler and 'scheduler' in state and state['scheduler']:
        print("[INFO] Loading aux_optimizer state_dict...")
        aux_optimizer.load_state_dict(state['aux_optimizer'])
        print("[INFO] Aux_optimizer state_dict loaded.")
        del state['aux_optimizer'] # Zwolnij RAM
        torch.cuda.empty_cache()
        gc.collect()
        aux_optimizer.zero_grad(set_to_none=True)

    # Zerowanie gradientów - dobra praktyka, ale nie powinny istnieć w tym momencie
    optimizer.zero_grad(set_to_none=True)

    if scheduler and 'scheduler' in state and state['scheduler']:
        print("[INFO] Loading scheduler state_dict...")
        scheduler.load_state_dict(state['scheduler'])
        del state['scheduler'] # Zwolnij RAM
        print("[INFO] Scheduler state_dict loaded.")

    if aux_scheduler and 'aux_scheduler' in state and state['aux_scheduler']:
        print("[INFO] Loading aux_scheduler state_dict...")
        aux_scheduler.load_state_dict(state['aux_scheduler'])
        del state['aux_scheduler'] # Zwolnij RAM
        print("[INFO] Aux_scheduler state_dict loaded.")

    print("[INFO] Loading scaler state_dict...")
    scaler.load_state_dict(state['scaler'])
    # UWAGA: Ta linijka jest podejrzana. load_state_dict powinno przywrócić
    # odpowiednią skalę. Ręczne ustawianie jej może być niepotrzebne lub błędne.
    # Skomentuj ją na razie, chyba że masz bardzo konkretny powód.
    # scaler._scale = torch.tensor(2. ** 8, device=device)
    print("[INFO] Scaler state_dict loaded.")
    del state['scaler'] # Zwolnij RAM
    gc.collect()

    start_epoch = state['epoch'] + 1

    # Pobierz zadanie ClearML dopiero po zwolnieniu pamięci przez tensory
    # (choć to raczej nie jest problemem pamięci GPU)
    task = Task.get_task(task_id=task_id)
    print(f"[INFO] Loaded ClearML Task: {task.name} (id={task_id})")


    print(f"[INFO] Successfully loaded training state from epoch {start_epoch - 1}")
    # Ostateczne czyszczenie
    del state
    torch.cuda.empty_cache()
    gc.collect()

    return task, start_epoch
