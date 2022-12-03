# Main .py file to run the project
from pathlib import Path
thispath = Path(__file__).resolve()
import sys

print(thispath.parent)
sys.path.insert(0, str(thispath.parent))

from database import SkinLesionDataset, metadata_creation

if __name__ == '__main__':
    metadata_creation('BinaryClassification')
    dataset = SkinLesionDataset('BinaryClassification', "train")

    print(dataset)