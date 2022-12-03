import torch
import torchvision.models as models


def model_option(model_name, num_classes, use_pretrained=True, freeze=False):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    net = None
    resize_param = 0
    if model_name == "resnet":
        """ ResNet50 
          """
        net = models.resnet50(pretrained=use_pretrained)
        num_ftrs = net.fc.in_features
        net.fc = torch.nn.Linear(num_ftrs, num_classes)
        resize_param = 224

    elif model_name == "convnext":
        """ ConvNeXt small
          """
        net = models.convnext_small(pretrained=use_pretrained)
        num_ftrs = net.classifier[2].in_features
        net.classifier[2] = torch.nn.Linear(num_ftrs, num_classes)
        if freeze:
            for param in net.features[:-2].parameters():
                param.requires_grad = False
        resize_param = 224

    elif model_name == "swim":
        """ Swim Transformer V2 -T
        """
        net = models.swin_v2_t(pretrained=use_pretrained)
        num_ftrs = net.head.in_features
        net.head = torch.nn.Linear(num_ftrs, out_features=num_classes)
        if freeze:
            for param in net.features[:-5].parameters():
                param.requires_grad = False
        resize_param = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return net, resize_param