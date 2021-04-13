import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wbf-file', type=str, default='data/input/nonR8R9R10_wbf.csv', help='path to csv with annotations processed by wbf')
    parser.add_argument('--data-dir', type=str, default='data', help = 'path to data directory')
    parser.add_argument('--img-size', type=int, default=512, help = '(square) dimensions of training images for aortic specialist')

    opt = parser.parse_args()


    DATA_DIR = opt.data_dir
    IMG_SIZE = opt.img_size

    class_mapping = {
        0: 0, #Aortic enlargement
        2: 1  #Calcification
    }


    for FOLD, EXP_NAME in enumerate(['exp_aortic_fold0', 'exp_aortic_fold1', 'exp_aortic_fold2', 'exp_aortic_fold3', 'exp_aortic_fold4']):

        print(f'Setting up yolo training files for {EXP_NAME}...')

        train_df = pd.read_csv(opt.wbf_file)

        train_df['x_mid'] = train_df.apply(lambda row: (row.x_max_norm+row.x_min_norm)/2, axis = 1)
        train_df['y_mid'] = train_df.apply(lambda row: (row.y_max_norm+row.y_min_norm)/2, axis = 1)

        train_df['w'] = train_df.apply(lambda row: (row.x_max_norm-row.x_min_norm), axis = 1)
        train_df['h'] = train_df.apply(lambda row: (row.y_max_norm-row.y_min_norm), axis = 1)

        train_df['x_min'] = train_df['x_min_norm'].copy()

        train_df = train_df.loc[train_df['class_id'].isin([0, 2])]
        train_df['class_id'] = train_df['class_id'].apply(lambda x: class_mapping[x])

        fold_valid_ids = np.load(os.path.join(DATA_DIR, f'validation_folds_v1/val_fold{FOLD}.npy'),
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
