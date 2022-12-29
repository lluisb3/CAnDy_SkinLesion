# Main .py file to run the project
from pathlib import Path
import torch
from matplotlib import pyplot as plt

thispath = Path(__file__).resolve()

import sys
from torchvision import transforms
from torch.utils.data import DataLoader
from classification_multi import model_option


sys.path.insert(0, str(thispath.parent))

from database import SkinLesionDataset


def show_image_batch(img_list):
    num = len(img_list)
    fig = plt.figure(figsize=(20,20))
    for i in range(num):
        ax = fig.add_subplot(1, num, i+1)
        img=img_list[i].numpy().transpose([1,2,0])
        ax.imshow(img)
    plt.show()


if __name__ == '__main__':
    net, resize_p = model_option('resnet', 3)
    DataAugmentation = transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(70),
                                                                   transforms.RandomVerticalFlip(),
                                                                   transforms.RandomHorizontalFlip(),
                                                                   transforms.RandomAffine(degrees=0, scale=(.9, 1.1),
                                                                                           translate=(0.2, 0.2),
                                                                                           shear=30),
                                                                   transforms.RandomPerspective(distortion_scale=0.3),
                                                                   transforms.GaussianBlur([3, 3])]), p=0.5)

    transform_train = transforms.Compose([transforms.ToTensor(), DataAugmentation, transforms.Resize(size=(resize_p, resize_p))])
    dataset_train = SkinLesionDataset('MulticlassClassification', "train", transform=transform_train)
    data_loader_train = DataLoader(dataset_train, batch_size=5)
    for i, minibatch in enumerate(data_loader_train):
        if i >= 2:
            break
        data = minibatch
        images, labels = data['image'], data['label']
        show_image_batch(images)
