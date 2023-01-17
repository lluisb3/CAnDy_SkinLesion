import torch
from pathlib import Path
import os
import click
import torchvision.transforms as transforms
from torchvision.io import read_image
from tqdm import tqdm
from segmentation import UNet, transform_segment, store

thispath = Path(__file__).resolve().parent.parent


def segment(dataset_option, subdataset_option):
    """
    Perform the segmentation of the Skin Lesion images of a desired dataset and sub-dataset.
    Parameters
    ----------
    dataset_option: Chose the Skin Lesion dataset, as "BinaryClassification" or "MulticlassClassification".
    subdataset_option: Chose the sub-dataset, as "train" (train) or "val" (validation).

    Returns
    -------
    In the folder dataset_option a folder called SUBDATASET_OPTION_seg saves all the segmentations contained in the
    subdataset_option organized in the same folder structure.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers = os.cpu_count() - 6

    print(f"Number of workers: {num_workers}")
    print(f"Device: {device}")

    modelsdir = Path(thispath / "models")
    model_filename = "Unet_trained.tar"

    net = UNet()
    checkpoint = torch.load(Path(modelsdir / model_filename), map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net_state_dict'])
    valid_jacc_net = checkpoint['accuracy']
    train_jacc_net = checkpoint['accuracy_train']

    print("Loaded pretrained U-net model for SkinLesion segmentation that reached jaccard of %.2f%% in validation and"
          " %.2f%% in training." % (max(valid_jacc_net), max(train_jacc_net)))

    net.to(device)
    net.eval()
    transform = transform_segment()

    datadir = Path(thispath / "data" / dataset_option / subdataset_option)

    patients_lesion_all_types = [x.stem for x in datadir.iterdir() if x.is_dir()]

    for patients_lesion_type in patients_lesion_all_types:

        lesion_path = [i for i in datadir.rglob("*.jpg") if patients_lesion_type in str(i)]

        # Do not accumulate gradients
        with torch.no_grad():

            # test all batches
            for image in tqdm(lesion_path, desc=patients_lesion_type):
                lesion_image = read_image(str(image))

                # Transform
                input_image = transform(lesion_image)
                input_image = torch.unsqueeze(input_image, dim=0)
                # Move data to device
                input_image = input_image.to(device, non_blocking=True)

                # Forward pass
                output = net(input_image)
                output_binary = torch.round(output)

                # Resize outputs to original size
                height = lesion_image.shape[1]
                width = lesion_image.shape[2]
                output_binary_original_size = transforms.functional.resize(output_binary, (height, width))

                # Store predictions
                name = image.stem
                output_dir = Path(datadir.parent / f"{subdataset_option}_seg" / patients_lesion_type)
                Path(output_dir).mkdir(exist_ok=True, parents=True)
                store(output_binary_original_size, name, output_dir)


@click.command()
@click.option(
    "--dataset_option",
    default="BinaryClassification",
    prompt="Choose the dataset to perform the segmentation with U-Net",
    help="Choose the dataset to perform the segmentation with U-Net",
)
@click.option(
    "--subdataset_option",
    default="train",
    prompt="Choose the subdataset, train (train) or val (validation)",
    help="Choose the subdataset to perform the segmentation with U-Net, train (train) or val (validation)",
)
def main(dataset_option, subdataset_option):
    # Perform the segmentations and save the results
    segment(dataset_option, subdataset_option)


if __name__ == "__main__":
    main()
