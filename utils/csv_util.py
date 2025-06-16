import csv
import datetime
import os
from pathlib import Path

import chardet

from utils.file_util import get_root_dir, get_encoding

rootdir = Path(get_root_dir())


def read_numeric_csv(file_path):
    data = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            numeric_row = []
            for cell in row:
                try:
                    # Try converting to float first (handles both int and float)
                    num = float(cell)
                    # Convert to int if it's a whole number for cleaner output
                    if num.is_integer():
                        num = int(num)
                    numeric_row.append(num)
                except ValueError:
                    raise ValueError(f"Non-numeric value found: '{cell}' in row: {row}")
            data.append(numeric_row)

    return data

def read(file_path,is_numeric=False):
    if is_numeric:
        return read_numeric_csv(file_path)

    data = []
    # 读取数据
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data


def read_df(abs_path=None, relative_path=None):
    import pandas as pd
    if abs_path:
        path = abs_path
    else:
        path = rootdir / relative_path

    df = pd.read_csv(path,encoding=get_encoding(path))
    # Display the dataframe
    return df

def is_2d(array):
    if not isinstance(array, (list, tuple)):
        return False
    return all(isinstance(row, (list, tuple)) for row in array) and len(array) > 0

def write_data(file_path, data):
    assert is_2d(data), "Data must be a 2D array (list of lists or tuples)"
    # 确保文件夹存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 写入数据
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def append_data(file_path, data):
    #make sure data is 2-d array
    #assert is_2d(data), "Data must be a 2D array (list of lists or tuples)"
     # 确保文件夹存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 追加数据
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        # Add empty line if file is not empty
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            writer.writerow([])  # Write empty row

        writer.writerows(data)


def test():
    # 写入数据
    data = [['Name', 'Age'], ['Alice', 25], ['Bob', 30]]
    write_data('d:/data.csv', data)

    # 追加数据
    new_data = [['Charlie', 35]]
    append_data('d:/data.csv', new_data)

    # 读取数据
    data = read('d:/data.csv')
    for row in data:
        print(row)

if __name__ == '__main__':

   # data =  read(is_numeric=True,file_path=r'D:\project\AiQLayout/results/evaluate/2025_05_09_10_32/2025_05_09_10_32.csv')
   # print(data)
   test()