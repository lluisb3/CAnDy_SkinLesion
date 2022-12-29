import torch
from torch.utils.data import DataLoader
from database import SkinLesionDataset
from torchvision import transforms
from utils import csv_writer
from pathlib import Path
import numpy as np

thispath = Path(__file__).resolve()


def get_mean_and_std(dataset_):
    channels_sum, channels_squared_sum = 0, 0
    for i in range(len(dataset_)):
        data = dataset_[i]
        channels_sum += data['image'].astype('float32').mean(axis=(0, 1))
        channels_squared_sum += (data['image'].astype('float32') ** 2).mean(axis=(0, 1))
    mean = channels_sum / len(dataset_)
    # std = sqrt(E[X^2] - (E[X])^2)
    std = ((channels_squared_sum / len(dataset_)) - (mean ** 2)) ** 0.5
    return list(mean), list(std)


if __name__ == '__main__':

    #transform_train = transforms.Compose([transforms.ToTensor()])
    train_dataset = SkinLesionDataset('MulticlassClassification', 'train')
    mean_train, std_train = get_mean_and_std(train_dataset)
    print('For the Multiclass Classification Challenge:')
    print(f'Mean: {mean_train}')
    print(f'Standard deviation:{std_train}')
    csv_writer(Path(thispath.parent.parent/'data'/'MulticlassClassification'), 'MulticlassClassification_statistics.csv',
               'w', ['Statistic', 'R', 'G', 'B'])
    row = ['Mean']
    row.extend(mean_train)
    csv_writer(Path(thispath.parent.parent / 'data' / 'MulticlassClassification'),
               'MulticlassClassification_statistics.csv',
               'a', row)
    row = ['Std']
    row.extend(std_train)
    csv_writer(Path(thispath.parent.parent / 'data' / 'MulticlassClassification'),
               'MulticlassClassification_statistics.csv',
               'a', row)
    train_dataset = SkinLesionDataset('BinaryClassification', 'train')
    mean_train, std_train = get_mean_and_std(train_dataset)
    print('For the Binary Classification Challenge:')
    print(f'Mean: {mean_train}')
    print(f'Standard deviation:{std_train}')
    csv_writer(Path(thispath.parent.parent / 'data' / 'BinaryClassification'),
               'BinaryClassification_statistics.csv',
               'w', ['Statistic', 'R', 'G', 'B'])
    row = ['Mean']
    row.extend(mean_train)
    csv_writer(Path(thispath.parent.parent / 'data' / 'BinaryClassification'),
               'BinaryClassification_statistics.csv',
               'a', row)
    row = ['Std']
    row.extend(std_train)
    csv_writer(Path(thispath.parent.parent / 'data' / 'BinaryClassification'),
               'BinaryClassification_statistics.csv',
               'a', row)
