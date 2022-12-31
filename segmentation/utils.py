from pathlib import Path
import numpy as np
import cv2 as cv
import torch
import torchvision.transforms as transforms


class Convert(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img)).float()


def transform_segment():
    # Resize image paramenter
    resize_param = [256, 320]

    # Train set statistics
    mu_train_r = 180.514
    mu_train_b = 139.454
    mu_train_g = 150.850
    std_train_r = 35.242
    std_train_b = 35.249
    std_train_g = 35.250

    # train set statics this project
    # mu_train_r=170.194
    # mu_train_g=134.993
    # mu_train_b=133.690
    # std_train_r=57.254
    # std_train_g=52.205
    # std_train_b=55.213

    transform = transforms.Compose([Convert(), transforms.Resize(size=resize_param),
                                    transforms.Normalize(mean=[mu_train_r, mu_train_g, mu_train_b],
                                                         std=[std_train_r, std_train_g, std_train_b])])

    return transform


def store(image, name, output_path):
    # path_images: where the images are to get the name
    # images: images you want to save
    # output_path: path to save the images
    image = torch.squeeze(image)
    image[image == 1] = 255
    name = f"{name}_seg.png"
    img_name = Path(output_path / name)
    cv.imwrite(str(img_name), image.cpu().numpy())
