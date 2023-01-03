import logging
import time
import random
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from database import SkinLesionDataset
from classification_multi import model_option
from torchvision import transforms

# from torch.utils.tensorboard import SummaryWriter

# ME ESTOY BASANDO EN https://github.com/joaco18/calc-det/blob/main/deep_learning/classification_models/train.py
# porfa checa las lineas    124,160
thispath = Path(__file__).resolve()
selected_gpu = 2  # here you select the GPU used (0, 1 or 2)
device = torch.device("cuda:" + str(selected_gpu) if
                      torch.cuda.is_available() else "cpu")
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', encoding='utf-8',
                    level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')


def train_1_epoch(net, train_dataset, train_dataloader, optimizer, criterion, scheduler):
    # switch to train mode
    net.train()
    # reset performance measures
    loss_sum, sample_counter = 0.0, 0
    predictions = torch.zeros((len(train_dataset), 1), dtype=torch.int64)
    labels = torch.zeros((len(train_dataset), 1), dtype=torch.int64)
    # 1 epoch = 1 complete loop over the dataset
    for batch in train_dataloader:
        # get data from dataloader
        inputs, targets = batch['image'], batch['label']
        # move data to device
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = net(inputs)
        # calculate loss
        loss = criterion(outputs, targets)
        # loss gradient backpropagation
        loss.backward()
        # net parameters update
        optimizer.step()
        # accumulate loss
        loss_sum += loss.item()

        # get labels
        outputs_max = torch.argmax(outputs, dim=1)

        # accumulate outputs and target
        for output, target in zip(outputs_max, targets):
            predictions[sample_counter] = output
            labels[sample_counter] = target
            sample_counter += 1
    # step learning rate scheduler once training all batches is done
    scheduler.step()
    # return average loss, predictions and ground truth labels
    return loss_sum / len(train_dataloader), predictions, labels


def val_1_epoch(net, val_dataset, val_dataloader, criterion):
    # switch to test mode
    net.eval()
    # initialize predictions
    predictions = torch.zeros((len(val_dataset), 1), dtype=torch.int64)
    labels = torch.zeros((len(val_dataset), 1), dtype=torch.int64)
    loss_sum, sample_counter = 0.0, 0
    # do not accumulate gradients (faster)
    with torch.no_grad():
        # 1 epoch = 1 complete loop over the dataset
        for batch in val_dataloader:
            # get data from dataloader
            inputs, targets = batch['image'], batch['label']
            # move data to device
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            # obtain predictions
            outputs = net(inputs)
            # calculate loss
            loss = criterion(outputs, targets)
            # accumulate loss
            loss_sum += loss.item()
            # store predictions
            outputs_max = torch.argmax(outputs, dim=1)
            for output, target in zip(outputs_max, targets):
                predictions[sample_counter] = output
                labels[sample_counter] = target
                sample_counter += 1

    return loss_sum / len(val_dataloader), predictions, labels


def train(net, skin_datasets, skin_dataloaders, criterion, optimizer, scheduler, cfg):
    # For logging
    exp_path = thispath.parent.parent / f'models/MulticlassClassification/{cfg["experiment_name"]}'
    exp_path.mkdir(exist_ok=True, parents=True)

    # For reproducibility
    since = time.time()
    random.seed(0)
    torch.manual_seed(1703)
    np.random.seed(0)

    # holders for best model
    best_metric = 0.0
    best_epoch = 0
    early_stopping_count = 0
    previous_loss = 0
    previous_mean_loss = 0
    best_metric_name = cfg['training']['best_metric']
    best_model_path = exp_path / f'{cfg["experiment_name"]}_{best_metric_name}.pt'
    chkpt_path = exp_path / f'{cfg["experiment_name"]}_chkpt.pt'
    logging.info(f'Storing experiment in: {exp_path}')

    if cfg['training']['resume_training']:
        checkpoint = torch.load(chkpt_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        init_epoch = checkpoint['epoch'] + 1
    else:
        init_epoch = 0

    # TENSORBOARD SETUP!!!
    # tensorboard logs
    # log_dir = exp_path / 'tensorboard'
    # log_dir.mkdir(exist_ok=True, parents=True)
    # writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(init_epoch, cfg['training']['n_epochs']):
        logging.info(f'Epoch {epoch + 1}/{cfg["training"]["n_epochs"]}')
        logging.info(('-' * 10))
        # Training the network
        avg_loss, net_predictions, gt_labels = train_1_epoch(net,
                                                             skin_datasets['train'],
                                                             skin_dataloaders['train'],
                                                             optimizer,
                                                             criterion,
                                                             scheduler
                                                             )
        # Get the metrics with the predictions and the labels
        # metrics_train = metrics_dl(net_predictions,gt_labels)
        # LOG THE METRICS INTO TENSORBOARD!
        ##### do something

        message = f"Epoch {epoch}: Train -- Avg Loss: {avg_loss:.4f} " \
                  f"Acc: {metrics_train['accuracy']:.4f} " \
                  f"BMA: {metrics_train['BMA']:.4f}, Kappa:{metrics_train['kappa']:.4f}"
        logging.info(message)

        # Validation of current network
        avg_loss, net_predictions, gt_labels = val_1_epoch(net,
                                                           skin_datasets['val'],
                                                           skin_dataloaders['val'],
                                                           criterion)
        # Get the metrics with the predictions and the labels
        # metrics_val = metrics_dl(net_predictions,gt_labels)
        # LOG THE METRICS INTO TENSORBOARD!
        ##### do something

        message = f"Epoch {epoch}: Val -- Avg Loss: {avg_loss:.4f} " \
                  f"Acc: {metrics_val['accuracy']:.4f} " \
                  f"BMA: {metrics_val['BMA']:.4f}, Kappa:{metrics_val['kappa']:.4f}"
        logging.info(message)

        # save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics_val': metrics_val,
            'metrics_train': metrics_train,
            'loss': avg_loss}, chkpt_path)

        # Save best checkpoint
        if metrics_val[best_metric_name] > best_metric:
            best_metric = metrics_val[best_metric_name]
            best_BMA = metrics_val['bma']
            best_acc = metrics_val['accuracy']
            best_epoch = epoch + 1
            torch.save({
                'model_state_dict': net.state_dict(),
                'metrics_val': metrics_val,
                'metrics_train': metrics_train,
                'configuration': cfg
            }, best_model_path)


def main():
    # read the configuration file
    config_path = str(thispath.parent / 'config.yml')
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    # use the configuration for the network
    model_arguments = cfg['model']
    net, resize_param = model_option(model_arguments['model_name'],
                                     model_arguments['num_classes'],
                                     freeze=model_arguments['freeze_weights'],
                                     num_freezed_layers=model_arguments['num_frozen_layers'])
    # Data transformations
    DataAugmentation = transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(70),
                                                                   transforms.RandomVerticalFlip(),
                                                                   transforms.RandomHorizontalFlip(),
                                                                   transforms.RandomAffine(degrees=0, scale=(.9, 1.1),
                                                                                           translate=(0.2, 0.2),
                                                                                           shear=30),
                                                                   transforms.GaussianBlur([3, 3])]),
                                              p=cfg['data_aug']['prob'])

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          DataAugmentation,
                                          transforms.Resize(size=(resize_param, resize_param))])
    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(size=(resize_param, resize_param))])

    # use the configuration for the dataset
    dataset_arguments = cfg['dataset']
    dataset_train = SkinLesionDataset(challenge_name=dataset_arguments['challenge_name'],
                                      dataset_set='train',
                                      dataset_mean=dataset_arguments['mean'],
                                      dataset_std=dataset_arguments['stddev'],
                                      transform=transform_train)
    dataset_val = SkinLesionDataset(challenge_name=dataset_arguments['challenge_name'],
                                    dataset_set='val',
                                    dataset_mean=dataset_arguments['mean'],
                                    dataset_std=dataset_arguments['stddev'],
                                    transform=transform_val)
    datasets = {'train': dataset_train, 'val': dataset_val}
    # use the configuration for the dataloader
    dataset_arguments = cfg['dataloaders']
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=dataset_arguments['train_batch_size'],
                                  shuffle=True,
                                  num_workers=dataset_arguments['num_workers'],
                                  pin_memory=True)
    dataloader_valid = DataLoader(dataset_val,
                                  batch_size=dataset_arguments['val_batch_size'],
                                  num_workers=dataset_arguments['num_workers'],
                                  pin_memory=True)
    dataloaders = {'train': dataloader_train, 'val': dataloader_valid}
    # loss function
    # Optimizer
    criterion = getattr(torch.nn, cfg['training']['criterion'])()

    optimizer = getattr(torch.optim, cfg['training']['optimizer'])
    optimizer = optimizer(net.parameters(), **cfg['training']['optimizer_args'])

    scheduler = getattr(torch.optim.lr_scheduler, cfg['training']['lr_scheduler'])
    scheduler = scheduler(optimizer, **cfg['training']['lr_scheduler_args'])
    # **d means "treat the key-value pairs in the dictionary as additional named arguments to this function call."

    train(net, datasets, dataloaders, criterion, optimizer, scheduler, cfg)


if __name__ == '__main__':
    main()
