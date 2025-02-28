from src.utils import initializers


class Experiment:
    def __init__(self, config):
        self.config = config

    def _init_from_config(self):
        self.dataset = initializers.dataset_from_config(self.config)
        self.model = initializers.model_from_config(self.config)
        self.loss = initializers.loss_from_config(self.config)
        self.optimizer = initializers.optimizer_from_config(self.model.parameters(), self.config)
        self.scheduler = initializers.scheduler_from_config(self.optimizer, self.config)