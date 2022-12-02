from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

thispath = Path(__file__).resolve()


class SkinLesionDataset(Dataset):
    def __init__(self, challenge_name, dataset_set, transform=None):
        self.data_dir = thispath.parent.parent / "data" / challenge_name / dataset_set
        self.metadata = pd.read_csv(self.data_dir.parent / f"Metadata_{challenge_name}.csv")
        self.transform = transform
        self.dataset_set = dataset_set

    def __len__(self):
        return self.metadata['Set'].value_counts().str(self.dataset_set)

    def __getitem__(self, idx):
        image = read_image(self.data_dir / f"{self.metadata.loc[idx, 'Name']}.jpg")
        label = self.metadata.loc[idx, 'Label']
        name = self.metadata.loc[idx, 'Name']

        seed = np.random.randint(654782)
        if self.transform:
            torch.manual_seed(seed)
            image = self.transform(image)

        return image, label, name
