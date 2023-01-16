
import numpy
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from database import SkinLesionDataset
from torchvision.utils import make_grid
from torchvision import transforms
from tqdm import tqdm
import click
import torchvision.transforms.functional as F
from classification_multi import model_option
from utils import csv_writer


thispath = Path(__file__).resolve()
selected_gpu = 2  # here you select the GPU used (0, 1 or 2)
device = torch.device("cuda:" + str(selected_gpu) if
                      torch.cuda.is_available() else "cpu")


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        print(img.shape)
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_image_batch(img_list):
    num = len(img_list)
    fig = plt.figure(figsize=(20, 20))
    for i in range(num):
        ax = fig.add_subplot(1, num, i + 1)
        img = img_list[i].numpy().transpose([1, 2, 0])
        ax.imshow(img)
    plt.show()


def test_binary(net, test_dataset, test_dataloader):
    # switch to test mode
    net.to(device)
    net.eval()
    # initialize predictions
    predictions = torch.zeros((len(test_dataset), 1), dtype=torch.int64)
    labels = torch.zeros((len(test_dataset), 1), dtype=torch.int64)
    sample_counter = 0
    with torch.no_grad():
        # 1 epoch = 1 complete loop over the dataset
        for batch in tqdm(test_dataloader, desc='Test'):
            # get data from dataloader
            inputs, targets = batch['image'], batch['label'].type(torch.LongTensor)
            # move data to device
            inputs = inputs.to(device, non_blocking=True)
            # obtain predictions
            outputs = net(inputs)
            outputs = torch.squeeze(outputs)
            outputs = torch.sigmoid(outputs)
            outputs_max = torch.round(outputs).int()
            for output, target in zip(outputs_max, targets):
                predictions[sample_counter] = output
                labels[sample_counter] = target
                sample_counter += 1
    return labels, predictions


def test_multiclass(net, test_dataset, test_dataloader):
    # switch to test mode
    net.to(device)
    net.eval()
    # initialize predictions
    predictions = torch.zeros((len(test_dataset), 1), dtype=torch.int64)
    labels = torch.zeros((len(test_dataset), 1), dtype=torch.int64)
    sample_counter = 0
    # do not accumulate gradients (faster)
    with torch.no_grad():
        # 1 epoch = 1 complete loop over the dataset
        for batch in tqdm(test_dataloader, desc='Test'):
            # get data from dataloader
            inputs, target = batch['image'], batch['label'].type(torch.LongTensor)
            # move data to device
            inputs = inputs.to(device, non_blocking=True)
            # obtain predictions
            outputs = net(inputs)
            # store predictions
            outputs_max = torch.argmax(outputs, dim=1)
            outputs_max = outputs_max.cpu().detach().numpy()

            for output, target in zip(outputs_max, targets):
                predictions[sample_counter] = output
                labels[sample_counter] = target
                sample_counter += 1
    print(f"{sample_counter} predictions saved on {thispath.parent}/final_predictions_multiclass.csv")
    return labels, predictions


# @click.command()
# @click.option(
#     "--challenge_name",
#     prompt="Choose the challenge to do the test",
#     help="Either MulticlassClassification or BinaryClassification",
# )
# @click.option(
#     "--trained_net",
#     prompt="Choose the trained network to use for test set for the challenge",
#     help="The directory should exist in models folder",
# )
def main(challenge_name, trained_net):
    if challenge_name != 'MulticlassClassification' and challenge_name != 'BinaryClassification':
        raise TypeError("Not a valid input argument, enter either 'MulticlassClassification' "
                        "or 'BinaryClassification'")

    print(f"Device: {device}")
    config_path = str(thispath.parent / 'models' / challenge_name / trained_net / f"config_{trained_net}.yml")
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    # use the configuration for the network
    model_arguments = cfg['model']
    net, resize_param = model_option(model_arguments['model_name'],
                                     model_arguments['num_classes'],
                                     freeze=model_arguments['freeze_weights'],
                                     num_freezed_layers=model_arguments['num_frozen_layers'],
                                     seg_mask=cfg['dataset']['use_masks'])
    # loading checkpoint
    best_metric_name = cfg['training']['best_metric']
    best_model = torch.load(
        thispath.parent / 'models' / challenge_name / trained_net / f"{trained_net}_{cfg['training']['best_metric']}.pt",
        map_location=lambda storage, loc: storage
    )
    net.load_state_dict(best_model['model_state_dict'])
    print(
        f"Loaded pretrained model {trained_net}... validation reached {best_metric_name} of {best_model['metrics_val'][best_metric_name]}")

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(size=(resize_param, resize_param))])
    # use the configuration for the dataset
    dataset_arguments = cfg['dataset']
    dataset_test = SkinLesionDataset(challenge_name=dataset_arguments['challenge_name'],
                                     dataset_set='val',
                                     dataset_mean=dataset_arguments['mean'],
                                     dataset_std=dataset_arguments['stddev'],
                                     transform=transform_test,
                                     seg_image=dataset_arguments['use_masks'])
    # use the configuration for the dataloader
    dataloader_arguments = cfg['dataloaders']
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=10,
                                 shuffle=False,
                                 num_workers=dataloader_arguments['num_workers'],
                                 pin_memory=True)
    for i, minibatch in enumerate(dataloader_test):
        if i >= 4:
            break
        data = minibatch
        std = torch.tensor(dataset_arguments['stddev']).view(1, 3, 1, 1) / 255
        mean = torch.tensor(dataset_arguments['mean']).view(1, 3, 1, 1) / 255
        images = data['image'][:, :3, :, :]
        print(images.shape)
        images = images * std + mean
        grid = make_grid(images)
        show(grid)
        plt.show()
    if challenge_name == 'MulticlassClassification':
        actual, predicted = test_multiclass(net, dataset_test, dataloader_test)
    elif challenge_name == 'BinaryClassification':
        actual, predicted = test_binary(net, dataset_test, dataloader_test)


if __name__ == '__main__':
    main('BinaryClassification', 'binary_resnet_unfreeze')





actual = numpy.random.binomial(1, .9, size=1000)
predicted = numpy.random.binomial(1, .9, size=1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

cm_display.plot()
plt.show()