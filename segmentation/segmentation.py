import torch
from pathlib import Path
import os
import click
import torchvision.transforms as transforms
from torchvision.io import read_image
from tqdm import tqdm
from segmentation import UNet, transform_segment, store

thispath = Path.cwd().resolve()


def segment(dataset_option):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers = os.cpu_count() - 6

    print(f"Number of workers: {num_workers}")
    print(f"Device: {device}")

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
    net.eval()
    transform = transform_segment()

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
            input_image = transform(image)
            input_image = torch.unsqueeze(input_image, dim=0)
            # Move data to device
            input_image = input_image.to(device, non_blocking=True)

            # Forward pass
            output = net(input_image)
            output_binary = torch.round(output)

            # Resize outputs to original size
            height = image.shape[1]
            width = image.shape[2]
            output_binary_original_size = transforms.functional.resize(output_binary, (height, width))

            # Store predictions
            name = nevus_image.stem
            output_dir = Path(datadir / "train_seg" / "nevus_prueba")
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            store(output_binary_original_size, name, output_dir)

        # test all batches
        for other_image in tqdm(others_path):
            image = read_image(str(other_image))

            # Transform
            input_image = transform(image)
            input_image = torch.unsqueeze(input_image, dim=0)

            # Move data to device
            input_image = input_image.to(device, non_blocking=True)

            # Forward pass
            output = net(input_image)
            output_binary = torch.round(output)

            # Resize outputs to original size
            height = image.shape[1]
            width = image.shape[2]
            output_binary_original_size = transforms.functional.resize(output_binary, (height, width))

            # Store predictions
            name = other_image.stem
            output_dir = Path(datadir / "train_seg" / "others_prueba")
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            store(output_binary_original_size, name, output_dir)


@click.command()
@click.option(
    "--dataset_option",
    default="BinaryClassification",
    help=
    "Chose the dataset to perform the segmentation with U-Net",
)
def main(dataset_option):
    # Perform the segmentations and save the results
    segment(dataset_option)


if __name__ == "__main__":
    main()
