import torch
import torchvision.models as models


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def model_option(model_name, num_classes, freeze=False):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    # if ever in need to delete cached weights go to Users\.cache\torch\hub\checkpoints
    net = None
    resize_param = 0
    if model_name == "resnet":
        """ ResNet50 
          """
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = net.fc.in_features
        net.fc = torch.nn.Linear(num_ftrs, num_classes)
        net_layers = [name for name, _ in net.named_children() if name != 'fc' and name != 'avgpool']

        resize_param = 224

    elif model_name == "convnext":
        """ ConvNeXt small
          """
        net = models.convnext_small(weights='DEFAULT')
        num_ftrs = net.classifier[2].in_features
        net.classifier[2] = torch.nn.Linear(num_ftrs, num_classes)
        net_layers = [name for name, _ in net.features.named_children()]

        if freeze:
            for param in net.features[:-2].parameters():
                param.requires_grad = False
        resize_param = 224

    elif model_name == "swim":
        """ Swim Transformer V2 -T
        """
        net = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
        num_ftrs = net.head.in_features
        net.head = torch.nn.Linear(num_ftrs, out_features=num_classes)
        net_layers = [name for name, _ in net.features.named_children()]
        if freeze:
            for param in net.features[:-5].parameters():
                param.requires_grad = False
        resize_param = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return net, resize_param
