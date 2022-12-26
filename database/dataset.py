from pathlib import Path
import torch
import numpy as np
import pandas as pd
import cv2 as cv
from utils import z_score_norm
from torchvision.io import read_image
from torch.utils.data import Dataset

thispath = Path(__file__).resolve()


class SkinLesionDataset(Dataset):
    def __init__(self, challenge_name, dataset_set, dataset_mean=None, dataset_std=None, transform=None):
        self.data_dir = thispath.parent.parent / "data" / challenge_name / dataset_set
        self.metadata = pd.read_csv(self.data_dir.parent / f"Metadata_{challenge_name}.csv")
        self.transform = transform
        self.mean=dataset_mean
        self.std=dataset_std
        self.dataset_set = dataset_set

    def __len__(self):
        return self.metadata['Set'].value_counts().loc[self.dataset_set]

    def __getitem__(self, idx):
        img = cv.imread(str(self.data_dir / f"{self.metadata['Lesion Type'].iloc[idx]}/"
                         f"{self.metadata['Name'].iloc[idx]}.jpg"))  # read the image (BGR) using OpenCV (HxWxC)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # image now RGB
        if self.transform:
            img = z_score_norm(img,
                               mean=self.mean,
                               std=self.std,
                               only_non_zero=self.metadata['FOV presence'].iloc[idx])  # Image in float-32 (HxWxC)
            img = self.transform(img)

        sample = {'image': img, 'label': self.metadata['Label'].iloc[idx]}
        return sample
