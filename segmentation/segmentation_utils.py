from pathlib import Path
import numpy as np
import cv2 as cv
import torch
import torchvision.transforms as transforms


class Convert(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img)).float()


def transform_segment():
    """
    Creates a transformation object ready to be applied to a given image to meet the requirements
    for the U-Net segmentation.
    Returns
    -------
    The transformation object with the necessary conversion toTensor, Resize and Normalization.
    """
    # Resize image parameter to network requisite input size
    resize_param = [256, 320]

    # Train set statistics for Normalization
    mu_train_r = 180.514
    mu_train_b = 139.454
    mu_train_g = 150.850
    std_train_r = 35.242
    std_train_b = 35.249
    std_train_g = 35.250

    transform = transforms.Compose([Convert(), transforms.Resize(size=resize_param),
                                    transforms.Normalize(mean=[mu_train_r, mu_train_g, mu_train_b],
                                                         std=[std_train_r, std_train_g, std_train_b])])

    return transform


def store(image, name, output_path):
    """

    Parameters
    ----------
    image: image to be stored
    name: name of the output image
    output_path: directory of the output image

    Returns
    -------
    Save the image in the desired folder
    """
    image = torch.squeeze(image)
    image[image == 1] = 255
    name = f"{name}_seg.png"
    img_name = Path(output_path / name)
    cv.imwrite(str(img_name), image.cpu().numpy())
