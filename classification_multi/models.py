import torch
from torch import nn
import torchvision.models as models


def set_parameter_requires_grad(model, number_frozen_layers):
    for k, child in enumerate(model.named_children()):
        if k == number_frozen_layers or k == 8:
            break
        for param in child[1].parameters():
            param.requires_grad = False
    return model


def model_option(model_name, num_classes, freeze=False, num_freezed_layers=0):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    # if ever in need to delete cached weights go to Users\.cache\torch\hub\checkpoints
    net = None
    resize_param = 0
    if model_name == "resnet":
        """ ResNet50 
          """
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if freeze:
            # Freezing the number of layers
            net = set_parameter_requires_grad(net, num_freezed_layers)

        num_ftrs = net.fc.in_features  # 2048
        net.fc = nn.Sequential(nn.Linear(num_ftrs, 500),
                               nn.ReLU(inplace=True),
                               nn.Linear(500, 50),
                               nn.ReLU(inplace=True),
                               nn.Linear(50, num_classes))
        resize_param = 224

    elif model_name == "convnext":
        """ ConvNeXt small
          """
        net = models.convnext_small(weights='DEFAULT')
        if freeze:
            # Freezing the number of layers
            net.features = set_parameter_requires_grad(net.features, num_freezed_layers)
        num_ftrs = net.classifier[2].in_features  # 768
        net.classifier[2] = nn.Sequential(nn.Linear(num_ftrs, 250),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(250, 50),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(50, num_classes))
        resize_param = 224

    elif model_name == "swin":
        """ Swin Transformer V2 -T
        """
        net = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
        if freeze:
            # Freezing the number of layers
            net = set_parameter_requires_grad(net, num_freezed_layers)
        num_ftrs = net.head.in_features  # 768
        net.head = nn.Sequential(nn.Linear(num_ftrs, 250),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(250, 50),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(50, num_classes))
        resize_param = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return net, resize_param
