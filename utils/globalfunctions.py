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


def get_fov(img):
    """
        This function returns a binary image with values =0 in the pixels with low intensities (as the FOV)
        :param img:  image with FOV
        ---------------
        :return thresh1: FOV with values of 0 the image with values of 1
    """
    copy_img = img.copy()
    height, width = copy_img[:, :, 1].shape
    gray_img= cv.cvtColor(copy_img, cv.COLOR_BGR2GRAY)
    top_left_corner = np.mean(gray_img[0:3, 0:3])
    top_right_corner = np.mean(gray_img[0:3, width - 3:width])
    bottom_left_corner = np.mean(gray_img[height - 3:height, 0:3])
    bottom_right_corner = np.mean(gray_img[height - 3:height, width - 3:width])

    # OTSU Thresholding if there is FOV also remove it
    if int(top_left_corner < 40) + int(top_right_corner < 40) + int(bottom_left_corner < 40) + int(
            bottom_right_corner < 40) > 2:
        # Internal function get_fov
        image_fov = get_fov(gray_img_copy)

    return thresh
