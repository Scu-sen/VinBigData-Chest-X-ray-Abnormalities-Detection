import os
import cv2
import numpy as np
import zipfile
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from aortic_util import read_xray
from joblib import Parallel, delayed
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, default='data/input/train', help='path to directory containing train/test dicoms')
    parser.add_argument('--img-size', type=int, default=512, help='(square) dimensions of final png images')
    parser.add_argument('--output-file', type=str, default='png_processed.zip', help='zip filename for converted images')
    opt = parser.parse_args()

    IMG_DIR = Path(opt.img_path)
    files = os.listdir(opt.img_path)
    IMAGE_SIZE = (opt.img_size, opt.img_size)
    OUT_FILE = opt.output_file

    x_tot,x2_tot = [],[]
    batch = 50

    read_xray_resize = lambda x: cv2.resize(read_xray(x), IMAGE_SIZE)

    with zipfile.ZipFile(OUT_FILE, 'w') as img_out:
        for idx in range(0,len(files),batch):
            names = [str(IMG_DIR)+'/'+x for x in files[idx:idx+batch]]
            out = Parallel(n_jobs=-1)(delayed(read_xray_resize)(i) for i in names)

            for s in range(len(out)):
                img = out[s]
                x_tot.append((img/255.0).mean())
                x2_tot.append(((img/255.0)**2).mean())
                name = names[s].split('/')[-1].split('.')[0]
                img = cv2.imencode('.png',img)[1]
                img_out.writestr(name + '.png', img)
