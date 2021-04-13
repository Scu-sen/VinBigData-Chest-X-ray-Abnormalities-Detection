# From https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image

import os
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
from aortic_util import read_xray
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', choices = ['train', 'test'], help='split (train or test) for which to extract image dimensions')
    parser.add_argument('--output-file', type=str, default='train_meta.csv', help='output filename')
    parser.add_argument('--kaggle-data-dir', type=str, default='data/input/', help='path to directory containing data files provided from competition website')
    opt = parser.parse_args()

    load_dir = os.path.join(opt.kaggle_data_dir, opt.split)
    image_id = []
    dim0 = []
    dim1 = []

    for file in tqdm(os.listdir(load_dir)):
        xray = read_xray(f'{load_dir}/{file}')
        image_id.append(file.replace('.dicom', ''))
        dim0.append(xray.shape[0])
        dim1.append(xray.shape[1])

    df = pd.DataFrame.from_dict({'image_id': image_id, 'height': dim0, 'width': dim1})
    df.to_csv(opt.output_file, index=False)
