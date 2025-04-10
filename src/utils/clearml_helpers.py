from clearml import OutputModel, Task, Logger
from clearml.task import TaskInstance

from src.train.experiment import Experiment


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