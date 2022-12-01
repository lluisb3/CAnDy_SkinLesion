import os
import torch
import numpy as np
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, mask_dir, image_dir, transform=None, target_ori_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.mask_dir = mask_dir
        self.img_dir = image_dir
        self.transform = transform
        self.target_ori_transform = target_ori_transform

    def __len__(self): #check the size of the dataset
        return len(self.img_labels)

    def __getitem__(self, idx): # read one image
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + '.jpg')
        mask_path = os.path.join(self.mask_dir, self.img_labels.iloc[idx, 0] + '_segmentation.png')
        image = read_image(img_path)
        mask_ori = read_image(mask_path)
        mask_ori[mask_ori == 255] = 1
        name = self.img_labels.iloc[idx, 0];

        seed = np.random.randint(651998) # make a seed with numpy generator
        if self.transform:
            torch.manual_seed(seed)
            image = self.transform(image)
        if self.target_ori_transform:
            torch.manual_seed(seed)
            mask_ori = self.target_ori_transform(mask_ori)
        return image, mask_ori, name