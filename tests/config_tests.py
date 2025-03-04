import unittest

from clearml import InputModel

from src.utils.const import PROJECT_ROOT, EXPERIMENTS_CONFIG_PATH
from src.utils.initializers import read_config, dataset_from_config, model_from_config, dataloader_from_config, \
    optimizer_from_config, scheduler_from_config


class MyTestCase(unittest.TestCase):
    # TODO; Write tests for various cases

    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

        example_config = read_config(f"{EXPERIMENTS_CONFIG_PATH}/example_experiment.yaml")
        model_config = example_config['model']
        dataset_config = example_config['dataset']
        dataloader_config = example_config['dataloader']
        optimizer_config = example_config['optimizer']
        scheduler_config = example_config['scheduler']

        model1 = InputModel.import_model(
            name='Cokolwiek',
            weights_url='file:///run/media/jakub/Dane/Studia/INZ/models/SWIN-T-IC_0.2.0.pth',
            framework='PyTorch'
        )
        model = model_from_config(model_config)
        dataset = dataset_from_config(dataset_config)
        dataloader = dataloader_from_config(dataset, dataloader_config)
        optimizer = optimizer_from_config(model.parameters(), optimizer_config)
        scheduler = scheduler_from_config(optimizer, scheduler_config)

if __name__ == '__main__':
    unittest.main()
