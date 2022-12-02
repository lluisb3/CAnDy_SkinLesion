import csv
from pathlib import Path


def csv_writer(filename, action, row):
    """
    :param filename: string
    csv file name with its path
    :param action: char
     Either 'w' to write a new csv file or 'a' to append a new row
    :param row: list
     Data to be appended to new row

    :return:
    """
    filename = Path(filename)
    with open(filename, action, encoding='UTF8', newline='') as f:  # 'a' to append row
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()
