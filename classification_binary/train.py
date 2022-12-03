import yaml
from pathlib import Path


def train():
    thispath = Path(__file__).resolve()

    # read the configuration file
    config_path = str(thispath.parent.parent / 'classification_binary/config.yml')
    print(config_path)
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # use the configuration for the dataset
    dataset_arguments = cfg['dataset']
    print(dataset_arguments)