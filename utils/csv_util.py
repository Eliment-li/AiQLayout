import csv
import datetime
import os
from pathlib import Path
from utils.file.file_util import get_root_dir,get_encoding
import chardet
rootdir = Path(get_root_dir())



def read(file_path):
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


def write_data(file_path, data):
    # 确保文件夹存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 写入数据
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def append_data(file_path, data):
     # 确保文件夹存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 追加数据
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
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

def demo():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H-%M')
    rootdir = get_root_dir()
    csv_path = rootdir / 'benchmark' / 'a-result' / formatted_datetime / '.csv'
    print(csv_path)
    write_data(csv_path,[['datetime', 'qasm', 'rl', 'qiskit', 'rl_qiskit', 'result', 'iter', 'remark', 'checkpoint'] ])
if __name__ == '__main__':

    read_df(relative_path=r'data/train_demo/ae40/env6.csv')