from pathlib import Path
import cv2 as cv
from utils import csv_writer
import numpy as np

thispath = Path(__file__).resolve()



def metadata_creation(challenge_name='MulticlassClassification'):
    """

    Parameters
    ----------
    challenge_name: string
        Either

    Returns
    -------

    """
    datadir = thispath.parent.parent/"data"/challenge_name
    images_files_train = [i for i in datadir.rglob("*.jpg") if "train" in str(i)]
    images_files_validation = [i for i in datadir.rglob("*.jpg") if "val" in str(i)]
    images_files_test = [i for i in datadir.rglob("*.jpg") if "test" in str(i)]
    header = ['Name', 'Dataset', 'Lesion Type', 'Class(numerical)', 'Image Height(row)',
              'Image Width(col)', 'DataType']
    csv_writer(datadir/'Metadata_SkinLesionChallenge2.csv', 'w', header)
    if
        lesion_type = ['mel', 'bcc', 'scc']
    else:
        lesion_type = ['nevus', 'others']
    for file in images_files_train:
        skin_lesion = cv.imread(str(file))
        row = [Path(file.stem), Path(file.parent.parent.stem), Path(file.parent.stem),
               lesion_type.index(str(Path(file.parent.stem))), skin_lesion.shape[0], skin_lesion.shape[1],
               np.min(skin_lesion), np.max(skin_lesion), skin_lesion.dtype]
        csv_writer(datadir/'Metadata_SkinLesionChallenge2.csv', 'a', row)

    for file in images_files_validation:
        skin_lesion = cv.imread(str(file))
        row = [Path(file.stem), Path(file.parent.parent.stem), Path(file.parent.stem),
               lesion_type.index(str(Path(file.parent.stem))), skin_lesion.shape[0], skin_lesion.shape[1],
               skin_lesion.dtype]
        csv_writer(datadir / 'Metadata_SkinLesionChallenge2.csv', 'a', row)

    for file in images_files_test:
        skin_lesion = cv.imread(str(file))
        row = [Path(file.stem), Path(file.parent.parent.stem), '-',
               '-', skin_lesion.shape[0], skin_lesion.shape[1], skin_lesion.dtype]
        csv_writer(datadir / 'Metadata_SkinLesionChallenge2.csv', 'a', row)

    print(f'Metadata csv created at {datadir}')
