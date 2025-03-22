from src.utils import initializers


class Experiment:
    # TODO: Docstring
    def __init__(self, config):
        self.config = config
        self._init_objects_from_config()
        self._set_training_data()
        self._set_experiment_metadata()

    def _init_objects_from_config(self):
        self.dataset = initializers.dataset_from_config(self.config['dataset'])
        self.model = initializers.model_from_config(self.config['model'])
        distortion_loss = initializers.loss_from_config(self.config['loss'])
        self.loss = distortion_loss
        try:
            if bool(self.config['rd_loss']['enabled']):
                self.loss = initializers.rd_loss_wrapper_from_config(distortion_loss, self.config['rd_loss'])
        except KeyError:
            pass
        self.optimizer = initializers.optimizer_from_config(self.model.parameters(), self.config['optimizer'])
        self.scheduler = initializers.scheduler_from_config(self.optimizer, self.config['scheduler'])

    def _set_training_data(self):
        self.epochs = self.config['epochs']
        self.dataset_split_ratio = self.config['dataset_split_ratio']

    def _set_experiment_metadata(self):
        self.project = self.config['project_name']
        self.task = self.config['task_name']
        self.comment = self.config['comment']

    def output_model_name(self):
        return f"{self.config['output']['model']['name']}_{self.config['output']['model']['version']}"