# Main .py file to run the project
import torch
from pathlib import Path
thispath = Path(__file__).resolve()
import sys
from classification_binary import train, model_option

sys.path.insert(0, str(thispath.parent))

from database import SkinLesionDataset, metadata_creation

if __name__ == '__main__':
    train()

    # Initialize the model for this run
    net, resize_param = model_option("swim", 2, use_pretrained=True)

    # Print the model we just instantiated
    print(net)

    # Test to check if selected network is giving the number of desired outputs
    im = torch.randn(1, 3, 224, 224)
    out = net(im)
    print(out)
