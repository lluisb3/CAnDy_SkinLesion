import torch
import yaml
from pathlib import Path
from database import SkinLesionDataset
from classification_multi import model_option


def train():
    thispath = Path(__file__).resolve()
    # read the configuration file
    config_path = str(thispath.parent/'config.yml')
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    # use the configuration for the dataset
    dataset_arguments = cfg['dataset']
    dataset_train = SkinLesionDataset(challenge_name=dataset_arguments['challenge_name'],
                                      dataset_set='train',
                                      dataset_mean=dataset_arguments['mean'],
                                      dataset_std=dataset_arguments['stddev'],
                                      transform=None)
