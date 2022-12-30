import torch
from pathlib import Path
import os
import torch.nn as nn
from collections import OrderedDict
import torchvision.transforms as transforms
import cv2 as cv
from torchvision.io import read_image
import numpy as np
from tqdm import tqdm

thispath = Path.cwd().resolve().parent

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# how many workers for fetching data
num_workers = os.cpu_count() - 6
monitor_display = True  # whether to display monitored performance plots
# Check number of workers and device
print(f"Number of workers: {num_workers}")
print(f"Device: {device}")

dropout = 0.1
# Resize image paramenter
resize_param = [256, 320]
# Activation function
activation = nn.ReLU()


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder5 = UNet._block(features * 8, features * 16, name="enc5")
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 16, features * 32, name="bottleneck")

        self.upconv5 = nn.ConvTranspose2d(features * 32, features * 16, kernel_size=2, stride=2)
        self.decoder5 = UNet._block((features * 16) * 2, features * 16, name="dec5")
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        bottleneck = self.bottleneck(self.pool5(enc5))

        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2,), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1,), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", activation),
                    (name + "drop1", nn.Dropout(dropout)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", activation),
                    (name + "drop2", nn.Dropout(dropout)),
                ]
            )
        )


def store(image, name, output_path):
    # path_images: where the images are to get the name
    # images: images you want to save
    # output_path: path to save the images
    image = torch.squeeze(image)
    image[image == 1] = 255
    name = f"{name}_seg.png"
    img_name = Path(output_path / name)
    cv.imwrite(str(img_name), image.cpu().numpy())


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


class Convert(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img)).float()


transform = transforms.Compose([Convert(), transforms.Resize(size=resize_param),
                                transforms.Normalize(mean=[mu_train_r, mu_train_g, mu_train_b],
                                                     std=[std_train_r, std_train_g, std_train_b])])


def segment():
    dataset_option = "BinaryClassification"

    # Switch to evaluation mode
    net.eval()

    datadir = Path(thispath / "data" / dataset_option)

    nevus_path = [i for i in datadir.rglob("*.jpg") if "train" in str(i)
                  and "nevus" in str(i)]
    others_path = [i for i in datadir.rglob("*.jpg") if "train" in str(i)
                   and "others" in str(i)]
    # Do not accumulate gradients
    with torch.no_grad():

        # test all batches
        for nevus_image in tqdm(nevus_path):

            image = read_image(str(nevus_image))

            # Transform
            input = transform(image)
            input = torch.unsqueeze(input, dim=0)
            # Move data to device
            input = input.to(device, non_blocking=True)

            # Forward pass
            output = net(input)
            output_binary = torch.round(output)

            # Resize outputs to original size
            height = image.shape[1]
            width = image.shape[2]
            output_binary_original_size = transforms.functional.resize(output_binary, (height, width))

            # Store predictions
            name = nevus_image.stem
            output_dir = Path(datadir / "train_seg" / "nevus")
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            store(output_binary_original_size, name, output_dir)

        # test all batches
        for other_image in tqdm(others_path):
            image = read_image(str(other_image))

            # Transform
            input = transform(image)
            input = torch.unsqueeze(input, dim=0)

            # Move data to device
            input = input.to(device, non_blocking=True)

            # Forward pass
            output = net(input)
            output_binary = torch.round(output)

            # Resize outputs to original size
            height = image.shape[1]
            width = image.shape[2]
            output_binary_original_size = transforms.functional.resize(output_binary, (height, width))

            # Store predictions
            name = other_image.stem
            output_dir = Path(datadir / "train_seg" / "others")
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            store(output_binary_original_size, name, output_dir)



modelsdir = Path(thispath / "models")

experiment_ID_load = "Unet_Experiment_7.tar"

net = UNet()

checkpoint = torch.load(Path(modelsdir / experiment_ID_load), map_location=lambda storage, loc: storage)

net.load_state_dict(checkpoint['net_state_dict'])
valid_jacc_net = checkpoint['accuracy']
train_jacc_net = checkpoint['accuracy_train']

print("Loaded pretrained U-net model for segmentation that reached jaccard of %.2f%% in validation and"
      " %.2f%% in training." % (max(valid_jacc_net), max(train_jacc_net)))

net.to(device)

# Perform the segmentations and save the results
segment()
