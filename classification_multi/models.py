import torch
from torch import nn
import torchvision.models as models
from torch.autograd import Variable

def set_parameter_requires_grad(model, number_frozen_layers):
    for k, child in enumerate(model.named_children()):
        if k == number_frozen_layers or k == 8:
            break
        for param in child[1].parameters():
            param.requires_grad = False
    return model


def model_option(model_name, num_classes, freeze=False, num_freezed_layers=0, seg_mask=False):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    # if ever in need to delete cached weights go to Users\.cache\torch\hub\checkpoints
    net = None
    resize_param = 0
    if model_name == "resnet":
        """ ResNet50 
          """
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if seg_mask:
            #   Modifying the input layer to receive 4-channel image instead of 3-channel image,
            #   We keep the pretrained weights for the RGB channels of the images
            weight1 = net.conv1.weight.clone()
            new_first_layer = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                        bias=False).requires_grad_()
            new_first_layer.weight[:, :3, :, :].data[...] = Variable(weight1, requires_grad=True)
            net.conv1 = new_first_layer

        if freeze:
            # Freezing the number of layers
            net = set_parameter_requires_grad(net, num_freezed_layers)

        num_ftrs = net.fc.in_features  # 2048
        net.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs // 4),
                               nn.ReLU(inplace=True),
                               nn.Linear(num_ftrs // 4, num_ftrs // 8),
                               nn.ReLU(inplace=True),
                               nn.Linear(num_ftrs // 8, num_classes))
        resize_param = 224

    elif model_name == "convnext":
        """ ConvNeXt small
          """
        net = models.convnext_small(weights='DEFAULT')
        if seg_mask:
            #   Modifying the input layer to receive 4-channel image instead of 3-channel image,
            #   We keep the pretrained weights for the RGB channels of the images
            weight1 = net.features[0][0].weight.clone()
            bias1 = net.features[0][0].bias.clone()
            new_first_layer = nn.Conv2d(4, 96, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0),
                                        bias=True).requires_grad_()
            new_first_layer.weight[:, :3, :, :].data[...] = Variable(weight1, requires_grad=True)
            new_first_layer.bias.data[...] = Variable(bias1, requires_grad=True)
            net.features[0][0] = new_first_layer
        if freeze:
            # Freezing the number of layers
            net.features = set_parameter_requires_grad(net.features, num_freezed_layers)
        num_ftrs = net.classifier[2].in_features  # 768
        net.classifier[2] = nn.Sequential(nn.Linear(num_ftrs, num_ftrs // 2),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(num_ftrs // 2, num_ftrs // 4),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(num_ftrs // 4, num_classes))
        resize_param = 224

    elif model_name == "swin":
        """ Swin Transformer V2 -T
        """
        net = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
        if seg_mask:
            #   Modifying the input layer to receive 4-channel image instead of 3-channel image,
            #   We keep the pretrained weights for the RGB channels of the images
            weight1 = net.features[0][0].weight.clone()
            bias1 = net.features[0][0].bias.clone()
            new_first_layer = nn.Conv2d(4, 96, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0),
                                        bias=True).requires_grad_()
            new_first_layer.weight[:, :3, :, :].data[...] = Variable(weight1, requires_grad=True)
            new_first_layer.bias.data[...] = Variable(bias1, requires_grad=True)
            net.features[0][0] = new_first_layer
        if freeze:
            # Freezing the number of layers
            net = set_parameter_requires_grad(net, num_freezed_layers)
        num_ftrs = net.head.in_features  # 768
        net.head = nn.Sequential(nn.Linear(num_ftrs, num_ftrs // 2),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(num_ftrs // 2, num_ftrs // 4),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(num_ftrs // 4, num_classes))
        resize_param = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return net, resize_param
