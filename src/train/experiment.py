import gc
import re
from typing import cast

import torch

from src.models.swin_compression import SwinTransformerCompressionAutoencoder
from src.utils import initializers


class Experiment:
    # TODO: Docstring
    def __init__(self, config):
        self.config = config
        self._init_objects_from_config()
        self._set_training_data()
        self._set_experiment_metadata()

    def _init_objects_from_config(self):
        self.train_dataset = initializers.dataset_from_config(self.config['train_dataset'])
        self.val_dataset = initializers.dataset_from_config(self.config['val_dataset'])
        self.test_dataset = initializers.dataset_from_config(self.config['test_dataset'])

        self.model: SwinTransformerCompressionAutoencoder = initializers.model_from_config(self.config['model'])
        distortion_loss = initializers.loss_from_config(self.config['loss'])
        self.loss = distortion_loss
        try:
            if bool(self.config['rd_loss']['enabled']):
                self.loss = initializers.rd_loss_wrapper_from_config(distortion_loss, self.config['rd_loss'])
        except KeyError:
            pass

        ordinary_params_ids = {id(p): p  for n, p in self.model.parameters(named=True)}
        conv_group_keys = {}
        for model in self.model.modules():
            if isinstance(model, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                for name, param in model.named_parameters():
                    if id(param) in ordinary_params_ids:
                        conv_group_keys[id(param)] = param

        remaining_group_keys = {k: v for k, v in ordinary_params_ids.items() if k not in conv_group_keys}
        self.param_groups = [
            {
                'params': remaining_group_keys.values(),
                'lr': self.config['optimizer']['main_lr']
            },
            {
                'params': conv_group_keys.values(),
                'lr': self.config['optimizer']['conv_lr']
            }
        ]
        self.optimizer = initializers.optimizer_from_config(self.param_groups, self.config['optimizer'])
        self.scheduler = initializers.scheduler_from_config(self.optimizer, self.config['scheduler'])
        self.aux_lr = self.config['aux_optimizer']['args']['lr']
        self.aux_optimizer = (
            initializers.optimizer_from_config(self.model.parameters(aux=True), self.config['aux_optimizer'])
            if not self.model.no_compress else None
        )
        self.aux_scheduler = (
            initializers.scheduler_from_config(self.aux_optimizer, self.config['aux_scheduler'])
            if not self.model.no_compress else None
        )

    def _set_training_data(self):
        self.epochs = self.config['epochs']
        self.dataset_split_ratio = int(self.config['dataset_split_ratio'])
        self.aux_optimizer_delay = int(self.config['aux_optimizer_delay'])
        self.log_frequency = int(self.config['log_frequency'])
        self.accumulation_steps = int(self.config['accumulation_steps'])

    def _set_experiment_metadata(self):
        self.project = self.config['project_name']
        self.task = self.config['task_name']
        self.comment = self.config['comment']

    def output_model_name(self):
        return f"{self.config['output']['model']['name']}_{self.config['output']['model']['version']}"