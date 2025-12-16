from clearml import OutputModel, Task, StorageManager
from clearml.task import TaskInstance
from src.train.experiment import Experiment
import numpy as np
import json
import os

from src.utils.const import TASKS_BACKUP_DIR


def save_model(task: TaskInstance, experiment: Experiment, weights_file_path: str):
    output_model = OutputModel(task=task, name=experiment.output_model_name())
    output_model.update_weights(weights_file_path)

    model_id = output_model.id
    print(f"Saved ClearML model with ID: {model_id}")

def start_experiment(experiment: Experiment) -> TaskInstance:
    task = Task.init(project_name=experiment.project, task_name=experiment.task)
    task.connect(experiment.config)
    task.set_comment(experiment.comment)
    return task

def backup_task(task_id: str):
    print(f"Backup task with ID: {task_id}")
    task_backup_dir = TASKS_BACKUP_DIR / task_id
    os.makedirs(task_backup_dir, exist_ok=True)
    task = Task.get_task(task_id=task_id)

    print("Backup task metadata")
    with open(f"{task_backup_dir}/task_metadata.json", "w") as f:
        json.dump(task.data.to_dict(), f, indent=2, default=str)

    print("Backup task parameters")
    params = task.get_parameters()
    with open(f"{task_backup_dir}/parameters.json", "w") as f:
        json.dump(params, f, indent=2, default=str)

    print("Backup configuration objects")
    configs = task.get_configuration_objects()
    with open(f"{task_backup_dir}/configurations.json", "w") as f:
        json.dump({k: v for k, v in configs.items()}, f, indent=2, default=str)

    print("Backup scalars")
    scalars = task.get_all_reported_scalars()
    with open(f"{task_backup_dir}/scalars.json", "w") as f:
        json.dump(scalars, f, indent=2, default=str)

    print("Backup console output")
    console = task.get_reported_console_output(number_of_reports=10_000)
    with open(f"{task_backup_dir}/console.log", "w") as f:
        f.write("\n".join(console))

    print("Backup completed:", task_backup_dir)

def clone_import_task(task_id: str):
    print(f"Clone task with ID: {task_id} and import data")
    task_backup_dir = TASKS_BACKUP_DIR / task_id
    new_task = Task.clone(task_id)
    log = new_task.get_logger()

    print("Import parameters")
    params = json.load(open(f"{task_backup_dir}/parameters.json"))
    for k, v in params.items():
        new_task.set_parameter(k, v)

    print("Import configuration objects")
    configs = json.load(open(f"{task_backup_dir}/configurations.json"))
    for name, value in configs.items():
        new_task.set_configuration_object(name, value)

    print("Import scalars")
    scalars = json.load(open(f"{task_backup_dir}/scalars.json"))
    for title, series_dict in scalars.items():
        for series, data in series_dict.items():
            for x, y in zip(data["x"], data["y"]):
                log.report_scalar(title, series, y, iteration=int(x))

    print("Import console output")
    console = open(f"{task_backup_dir}/console.log").read().split("\n")
    for chunk in console:
        log.report_text(chunk)

    new_task.close()

if __name__ == "__main__":
    task_id = "79180c9521ed435c859d0ede8a905c74"
    # backup_task(task_id)
    # clone_import_task(task_id)
    task = Task.get_task(task_id=task_id)
    task.mark_started(force=True)
    print(f"[INFO] Loaded ClearML Task: {task.name} (id={task_id})")
    logger = task.get_logger()
    logger.report_scalar(iteration=169600, title="PSNR", value=32.0297, series="train")

    task.mark_completed(force=True)
    task.close()
