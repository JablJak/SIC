import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MODEL_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoint"
MODEL_CHECKPOINT_FILE = "model.pth"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models"
EXPERIMENTS_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments"
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
TASKS_BACKUP_DIR = PROJECT_ROOT / "tasks"