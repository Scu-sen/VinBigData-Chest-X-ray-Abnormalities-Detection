import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default = 'data', help = 'path to data directory')

    opt = parser.parse_args()
    DATA_DIR = opt.data_dir

    for EXP_NAME, IMG_SIZE in zip(['exp_R8_512', 'exp_R8_1024'], [512, 1024]):

        print(f'Setting up yolo training files for {EXP_NAME}...')
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'input/train.csv'))
        train_meta = pd.read_csv(os.path.join(DATA_DIR, 'input/train_meta.csv'))

        train_df = train_df.merge(train_meta, how = 'left', on = 'image_id')
        train_df = train_df.loc[train_df['class_name'] != 'No finding']
        train_df = train_df.loc[train_df['rad_id'] == 'R8'].reset_index(drop = True)

        train_df['x_min'] = train_df.apply(lambda row: (row.x_min)/row.width, axis =1)
        train_df['y_min'] = train_df.apply(lambda row: (row.y_min)/row.height, axis =1)

        train_df['x_max'] = train_df.apply(lambda row: (row.x_max)/row.width, axis =1)
        train_df['y_max'] = train_df.apply(lambda row: (row.y_max)/row.height, axis =1)

        train_df['x_mid'] = train_df.apply(lambda row: (row.x_max+row.x_min)/2, axis =1)
        train_df['y_mid'] = train_df.apply(lambda row: (row.y_max+row.y_min)/2, axis =1)

        train_df['w'] = train_df.apply(lambda row: (row.x_max-row.x_min), axis =1)
        train_df['h'] = train_df.apply(lambda row: (row.y_max-row.y_min), axis =1)


        fold_valid_ids = np.load(os.path.join(DATA_DIR, 'validation_folds_v1/val_fold1.npy'),
                                 allow_pickle = True)

        train_df_subset = train_df.loc[~(train_df['image_id'].isin(fold_valid_ids))].reset_index(drop = True)
        valid_df_subset = train_df.loc[train_df['image_id'].isin(fold_valid_ids)].reset_index(drop = True)

        if os.path.isdir(os.path.join(f'{DATA_DIR}/yolo/yolo_experiment_data/{EXP_NAME}/')):
            shutil.rmtree(os.path.join(f'{DATA_DIR}/yolo/yolo_experiment_data/{EXP_NAME}/'))

        os.makedirs(os.path.join(f'{DATA_DIR}/yolo/yolo_experiment_data/{EXP_NAME}/labels/train'), exist_ok = True)
        os.makedirs(os.path.join(f'{DATA_DIR}/yolo/yolo_experiment_data/{EXP_NAME}/labels/val'), exist_ok = True)
        os.makedirs(os.path.join(f'{DATA_DIR}/yolo/yolo_experiment_data/{EXP_NAME}/images/train'), exist_ok = True)
        os.makedirs(os.path.join(f'{DATA_DIR}/yolo/yolo_experiment_data/{EXP_NAME}/images/val'), exist_ok = True)

        for image_id in tqdm(train_df_subset.image_id.unique()):

            tmp = train_df_subset.loc[(train_df_subset['image_id'] == image_id)]

            if tmp.shape[0] == 0:
                continue

            fname = f'{image_id}.txt'

            arr = np.array(tmp[['class_id', 'x_mid', 'y_mid', 'w', 'h']])

            np.savetxt(os.path.join(f'{DATA_DIR}/yolo/yolo_experiment_data/{EXP_NAME}/labels/train/', fname), arr, fmt='%5.4g')


        for image_id in tqdm(valid_df_subset.image_id.unique()):

            tmp = valid_df_subset.loc[(valid_df_subset['image_id'] == image_id)]

            if tmp.shape[0] == 0:
                continue

            fname = f'{image_id}.txt'

            arr = np.array(tmp[['class_id', 'x_mid', 'y_mid', 'w', 'h']])

            np.savetxt(os.path.join(f'{DATA_DIR}/yolo/yolo_experiment_data/{EXP_NAME}/labels/val/', fname), arr, fmt='%5.4g')


        # Get list of training image paths
        train_files = [f'{DATA_DIR}/extracted_images/{IMG_SIZE}/train/{id_}.png' for id_ in train_df_subset['image_id'].unique()]
        val_files   = [f'{DATA_DIR}/extracted_images/{IMG_SIZE}/train/{id_}.png' for id_ in valid_df_subset['image_id'].unique()]


        for file in tqdm(train_files):
            shutil.copy(file, f'{DATA_DIR}/yolo/yolo_experiment_data/{EXP_NAME}/images/train')
            filename = file.split('/')[-1].split('.')[0]

        for file in tqdm(val_files):
            shutil.copy(file, f'{DATA_DIR}/yolo/yolo_experiment_data/{EXP_NAME}/images/val')
            filename = file.split('/')[-1].split('.')[0]

    print('Done!')
