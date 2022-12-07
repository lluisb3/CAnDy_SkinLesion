from pathlib import Path
import cv2 as cv
from utils import csv_writer, check_fov
import numpy as np
from tqdm import tqdm

thispath = Path(__file__).resolve()


def metadata_creation(challenge_name):
    """

    Parameters
    ----------
    challenge_name (string): Either MulticlassClassification or BinaryClassification

    Returns
    -------
    Creates a csv file in the directory of the corresponding classification challenge
    """

    if challenge_name != 'MulticlassClassification' and challenge_name != 'BinaryClassification':
        raise TypeError("Not a valid input argument, enter either 'MulticlassClassification' "
                        "or 'BinaryClassification'")

    datadir = thispath.parent.parent/"data"/challenge_name
    images_files_train = [i for i in datadir.rglob("*.jpg") if "train" in str(i)]
    images_files_validation = [i for i in datadir.rglob("*.jpg") if "val" in str(i)]
    images_files_test = [i for i in datadir.rglob("*.jpg") if "test" in str(i)]

    if challenge_name == 'MulticlassClassification':
        header = ['Name', 'Set', 'Lesion Type', 'Label', 'Image Height(row)',
                  'Image Width(col)', 'Min', 'Max', 'FOV presence', 'DataType']
        lesion_type = ['mel', 'bcc', 'scc']

    else:
        header = ['Name', 'Set', 'Lesion Type', 'Lesion Subtype', 'Label',
                  'Image Height(row)', 'Image Width(col)', 'Min', 'Max', 'FOV presence', 'DataType']
        lesion_type = ['nevus', 'others']

    csv_writer(datadir, f'Metadata_{challenge_name}.csv', 'w', header)

    for i, file in zip(tqdm(range(len(images_files_train)), desc='Train files'), images_files_train):
        skin_lesion = cv.imread(str(file))
        row = [file.stem, file.parent.parent.stem, file.parent.stem,
               lesion_type.index(str(file.parent.stem)), skin_lesion.shape[0], skin_lesion.shape[1],
               np.min(skin_lesion), np.max(skin_lesion), check_fov(skin_lesion), skin_lesion.dtype]
        if challenge_name == 'BinaryClassification':
            row.insert(3, file.stem[:3])
        csv_writer(datadir, f'Metadata_{challenge_name}.csv', 'a', row)

    for i, file in zip(tqdm(range(len(images_files_validation)), desc='Validation files'), images_files_validation):
        skin_lesion = cv.imread(str(file))
        row = [file.stem, file.parent.parent.stem, file.parent.stem,
               lesion_type.index(str(file.parent.stem)), skin_lesion.shape[0], skin_lesion.shape[1],
               np.min(skin_lesion), np.max(skin_lesion), check_fov(skin_lesion), skin_lesion.dtype]
        if challenge_name == 'BinaryClassification':
            row.insert(3, file.stem[:3])
        csv_writer(datadir, f'Metadata_{challenge_name}.csv', 'a', row)

    for i, file in zip(tqdm(range(len(images_files_test)), desc='Test files'), images_files_test):
        skin_lesion = cv.imread(str(file))
        row = [file.stem, file.parent.parent.stem,'-', '-',
               skin_lesion.shape[0], skin_lesion.shape[1], np.min(skin_lesion), np.max(skin_lesion),
               check_fov(skin_lesion), skin_lesion.dtype]
        if challenge_name == 'BinaryClassification':
            row.insert(3, '-')
        csv_writer(datadir, f'Metadata_{challenge_name}.csv', 'a', row)

    print(f'Metadata csv created at {datadir}')