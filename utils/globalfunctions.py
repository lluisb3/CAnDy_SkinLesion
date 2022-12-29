import csv
import cv2 as cv
import numpy as np


def csv_writer(file_path, name, action, data):
    """

    Parameters
    ----------
    file_path (Path from pathlib): path where to save the csv file
    name (string): csv name
    action (char): Either 'w' to write a new csv file or 'a' to append a new row
    data (list): Data to be appended to new row

    Returns
    -------
    """
    absolute_path = file_path / name
    with open(absolute_path, action, encoding='UTF8', newline='') as f:  # 'a' to append row
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


def check_fov(img, threshold=40):
    """

    Parameters
    ----------
    img (numpy): Image data
    threshold (int): threshold to detect the fov

    Returns
    -------
    answer (bool): True if there is FOV False if not
    """
    copy_img = img.copy()
    height, width, _ = copy_img.shape
    gray_img = cv.cvtColor(copy_img, cv.COLOR_BGR2GRAY)
    top_left_corner = np.mean(gray_img[0:5, 0:5])
    top_right_corner = np.mean(gray_img[0:5, width - 3:width])
    bottom_left_corner = np.mean(gray_img[height - 5:height, 0:5])
    bottom_right_corner = np.mean(gray_img[height - 5:height, width - 5:width])


    # Check if there is FOV in at least 3 corners
    return int(top_left_corner < threshold) + int(top_right_corner < threshold) + int(bottom_left_corner < threshold) \
           + int(bottom_right_corner < threshold) > 2


def get_fov(img):
    """
        This function returns a binary image with values =0 in the pixels with low intensities (as the FOV)
        :param img:  image with FOV
        ---------------
        :return thresh1: FOV with values of 0, the image with values of 1
    """
    copy = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(copy, 15, 255, cv.THRESH_BINARY)
    kernel = np.ones((35, 35), np.uint8)
    tresh = cv.dilate(thresh,kernel,iterations = 3)
    thresh[thresh == 255] = 1

    return thresh


def z_score_norm(img, mean=None, std=None, only_non_zero=False):
    if mean is None:
        if only_non_zero:
            mask = get_fov(np.asarray(img))
            imgcopy = img.copy().astype('float32')
            imgcopy[mask == 0] = np.nan
            mean = np.nanmean(imgcopy, axis=(0, 1))
        else:
            mean = img.mean(axis=(0, 1))
    if std is None:
        if only_non_zero:
            mask = get_fov(img)
            imgcopy = img.copy().astype('float32')
            imgcopy[mask == 0] = np.nan
            std = np.nanstd(imgcopy, axis=(0, 1))
        else:
            std = img.std(axis=(0, 1))
    img = (img - mean) / std
    return img.astype('float32')