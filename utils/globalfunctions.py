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
    img (numpy) Image data
    threshold (int) threshold to detect the fov

    Returns
    -------
    answer (bool) True if there is FOV False if not
    """
    copy_img = img.copy()
    height, width, _ = copy_img.shape
    gray_img = cv.cvtColor(copy_img, cv.COLOR_BGR2GRAY)
    top_left_corner = np.mean(gray_img[0:5, 0:5])
    top_right_corner = np.mean(gray_img[0:5, width - 3:width])
    bottom_left_corner = np.mean(gray_img[height - 5:height, 0:5])
    bottom_right_corner = np.mean(gray_img[height - 5:height, width - 5:width])

    # Check if there is FOV in at least 3 corners
    if int(top_left_corner < threshold) + int(top_right_corner < threshold) + int(bottom_left_corner < threshold)\
            + int(bottom_right_corner < threshold) > 2:
        answer = True
    else:
        answer = False

    return answer
