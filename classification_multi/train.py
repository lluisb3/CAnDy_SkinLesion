import torch
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from database import SkinLesionDataset
from classification_multi import model_option
from torchvision import transforms


def train(net, dataset_skinlesion, dataloader_skinlesion):
    net.train()


def main():
    # For the gpu
    selected_gpu = 2  # here you select the GPU used (0, 1 or 2)
    device = torch.device("cuda:" + str(selected_gpu) if
                          torch.cuda.is_available() else "cpu")
    thispath = Path(__file__).resolve()
    # read the configuration file
    config_path = str(thispath.parent / 'config.yml')
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    # use the configuration for the network
    model_arguments = cfg['model']
    net, resize_param = model_option(model_arguments['model_name'],
                                     model_arguments['num_classes'],
                                     freeze=model_arguments['freeze_weights'])
    # Data transformations
    DataAugmentation = transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(70),
                                                                   transforms.RandomVerticalFlip(),
                                                                   transforms.RandomHorizontalFlip(),
                                                                   transforms.RandomAffine(degrees=0, scale=(.9, 1.1),
                                                                                           translate=(0.2, 0.2),
                                                                                           shear=30),
                                                                   transforms.GaussianBlur([3, 3])]),
                                              p=cfg['data_aug']['prob'])

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          DataAugmentation,
                                          transforms.Resize(size=(resize_param, resize_param))])
    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(resize_param, resize_param))])

    # use the configuration for the dataset
    dataset_arguments = cfg['dataset']
    dataset_train = SkinLesionDataset(challenge_name=dataset_arguments['challenge_name'],
                                      dataset_set='train',
                                      dataset_mean=dataset_arguments['mean'],
                                      dataset_std=dataset_arguments['stddev'],
                                      transform=transform_train)
    dataset_val = SkinLesionDataset(challenge_name=dataset_arguments['challenge_name'],
                                    dataset_set='val',
                                    dataset_mean=dataset_arguments['mean'],
                                    dataset_std=dataset_arguments['stddev'],
                                    transform=transform_val)
    # use the configuration for the dataloader
    dataset_arguments = cfg['dataloaders']
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=dataset_arguments['train_batch_size'],
                                  shuffle=True,
                                  num_workers=dataset_arguments['num_workers'],
                                  pin_memory=True)
    dataloader_valid = DataLoader(dataset_val,
                                  batch_size=dataset_arguments['val_batch_size'],
                                  num_workers=dataset_arguments['num_workers'],
                                  pin_memory=True)
