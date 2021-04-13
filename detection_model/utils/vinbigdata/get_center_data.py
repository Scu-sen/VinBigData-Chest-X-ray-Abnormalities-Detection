"""
According to the idea of Senyang, calculate the mean and std statistics of each center, and then
normalize the images for each of the centers, using these means and stds.
"""
import pandas as pd
import os
import os.path as osp
import mmcv
import glob
import pydicom
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count


def concact_all_center_csvs(csv_dir, save_file=None):
    """
    Return:
        df_center: image_id    center_id
        number of centers
    """
    csv_files = glob.glob(os.path.join(csv_dir, 'train_*.csv'))
    csv_files = sorted(csv_files)
    print('num_files: ', len(csv_files))
    # to ensure the center are the same for every runs
    dfs = [pd.read_csv(file) for file in csv_files]

    all_img_ids = []
    all_center_ids = []
    for ct_idx, df in enumerate(dfs):
        cts = [ct_idx for _ in range(df.shape[0])]
        img_ids = df['file'].tolist()
        all_img_ids.extend(img_ids)
        all_center_ids.extend(cts)
    df = pd.DataFrame()
    df['image_id'] = all_img_ids
    df['center_id'] = all_center_ids

    if save_file is not None:
        df.to_csv(save_file, index=False)

    assert len(all_img_ids) == 15000, 'number of images {}'.format(len(all_img_ids))
    return df, ['_'.join(osp.splitext(osp.basename(file))[0].split('_')[1:]) for file in csv_files]


def convert_one_dicom(src_file, out_file):
    dicom = pydicom.read_file(src_file)
    data = dicom.pixel_array
    np.save(out_file, data)
    return True


def dicom_to_numpy_array(src_path, out_path, json_file, multiprocessing=False, convert_num=None):
    """
    convert the dicom image to numpy array. Only transform abnormal images.
    Args:
        src_path (str): path to load dicom image.
        out_path (str): path to save numpy image
        json_file (str): path to load the files that we need to transform
    """
    mmcv.mkdir_or_exist(out_path)

    json_infos = mmcv.load(json_file)
    if len(json_infos['annotations']) > 10:  # train / val split
        image_boxes = defaultdict(list)
        for anno in json_infos['annotations']:
            image_boxes[anno['image_id']].append(anno['category_id'])
        imgId2name = {info['id']: os.path.splitext(info['file_name'])[0]
                      for info in json_infos['images']}
        image_names = []
        for im_id, labels in image_boxes.items():
            if len(labels) == 1 and labels[0] == 15:  # no annotations
                continue
            image_names.append(imgId2name[im_id])
    else:   # test split
        image_names = [os.path.splitext(info['file_name'])[0] for info in json_infos['images']]

    if convert_num is not None:
        assert isinstance(convert_num, int)
        image_names = image_names[:convert_num]

    print('There are {} images in total'.format(len(image_names)))
    print('start converting images...')
    if multiprocessing:
        pools = Pool(cpu_count())
        async_results = []
        for img_id in image_names:
            async_results.append(
                pools.apply_async(convert_one_dicom,
                                  (os.path.join(src_path, img_id + '.dicom'),
                                   os.path.join(out_path, '{}.npy'.format(img_id)))))
        for async_res in tqdm(async_results, total=len(async_results)):
            async_res.wait()
            res = async_res.get()
    else:
        for img_id in tqdm(image_names, total=len(image_names)):
            convert_one_dicom(
                src_file=os.path.join(src_path, img_id + '.dicom'),
                out_file=os.path.join(out_path, img_id + '.npy'))

    print('convert done!!')


def get_data_mean(file):
    data = pydicom.read_file(file).pixel_array
    num_pixel = data.shape[0] * data.shape[1]
    sum_pixel = np.sum(data)
    return sum_pixel, num_pixel


def get_data_std(file, mean):
    data = pydicom.read_file(file).pixel_array
    std_value = np.sum((data - mean) ** 2)
    return std_value


def get_ceneter_mean_std_mulpro(data_path, center_csv_file, save_file, num_center=13):
    """Multi-processing version of getting mean and std"""
    df = pd.read_csv(center_csv_file)
    df['mean'] = [0] * len(df)
    df['std'] = [1] * len(df)
    pools = Pool(cpu_count())
    for ct in tqdm(range(num_center)):  # loop through centers
        image_ids = df[df['center_id'] == ct]['image_id'].tolist()

        # 1. calculate mean value
        async_results = []
        img_mean_infos = []
        for img_id in image_ids:
            async_results.append(
                pools.apply_async(get_data_mean,
                                  (os.path.join(data_path, img_id + '.dicom'), )
                                  ))
        for async_res in tqdm(async_results, total=len(async_results)):
            async_res.wait()
            res = async_res.get()
            img_mean_infos.append(res)

        num_pixels = sum(item[1] for item in img_mean_infos)
        mean_value = sum(item[0] for item in img_mean_infos)
        mean_value /= num_pixels
        del img_mean_infos

        # 1. calculate std value
        async_results = []
        img_std_infos = []
        for img_id in image_ids:
            async_results.append(
                pools.apply_async(get_data_std,
                                  (os.path.join(data_path, img_id + '.dicom'), mean_value)
                                  ))
        for async_res in tqdm(async_results, total=len(async_results)):
            async_res.wait()
            res = async_res.get()
            img_std_infos.append(res)
        std_value = sum(img_std_infos)
        std_value = np.sqrt(std_value / num_pixels)

        all_means = [mean_value] * len(image_ids)
        all_stds = [std_value] * len(image_ids)
        df.loc[df['center_id'] == ct, 'mean'] = all_means
        df.loc[df['center_id'] == ct, 'std'] = all_stds
    df.to_csv(save_file, index=False)
    print('all done!')


def get_ceneter_mean_std(data_path, center_csv_file, save_file, num_center=13):
    """get the mean and std of each center data."""
    df = pd.read_csv(center_csv_file)
    df['mean'] = [0] * len(df)
    df['std'] = [1] * len(df)
    for ct in tqdm(range(num_center)):  # loop through centers
        df_ct = df[df['center_id'] == ct]
        image_ids = df_ct['image_id'].tolist()
        num_pixel = 0
        mean_value = 0
        std_value = 0
        for im_id in tqdm(image_ids):  # calculate mean, loop through images in one center
            file = osp.join(data_path, im_id + '.dicom')
            data = pydicom.read_file(file).pixel_array
            num_pixel += data.shape[0] * data.shape[1]
            mean_value += np.sum(data)
        mean_value = mean_value / num_pixel

        for im_id in tqdm(image_ids):  # calculate std, loop through images in one center
            file = osp.join(data_path, im_id + '.dicom')
            data = pydicom.read_file(file).pixel_array
            std_value += np.sum((data - mean_value) ** 2)
        std_value = np.sqrt(std_value / num_pixel)

        all_means = [mean_value] * len(image_ids)
        all_stds = [std_value] * len(image_ids)
        df.loc[df['center_id'] == ct, 'mean'] = all_means
        df.loc[df['center_id'] == ct, 'std'] = all_stds
    df.to_csv(save_file, index=False)
    print('all done!')


def add_mean_std_to_jsonAnno(src_anno_file, out_file, csv_info_file):
    """
    Add mean and std statistics into the original json annotation file.
    image: {
        'id': int,
        'file_name': .png --> .npy
        'mean': float,
        'std': float,
        ...
    }
    Args:
        src_anno_file (str): path to load original annotation file
        out_file (str): path to save the new annotations
        csv_info_file (str): path to load the mean and std statistics of each image
    """
    mmcv.mkdir_or_exist(osp.dirname(out_file))
    all_annotations = mmcv.load(src_anno_file)
    df_info = pd.read_csv(csv_info_file)
    imgId2info = {}
    for idx, row in df_info.iterrows():
        img_id = row['image_id']
        imgId2info[img_id] = (row['mean'], row['std'])

    images = all_annotations['images']
    for idx in tqdm(range(len(images))):
        img_info = images[idx]
        img_id = osp.splitext(img_info['file_name'])[0]
        mean, std = imgId2info[img_id]
        img_info['mean'] = mean
        img_info['std'] = std
        img_info['file_name'] = img_id + '.npy'
        images[idx] = img_info

    print('add done!\nsaving new annotations into file...')
    all_annotations['images'] = images
    mmcv.dump(all_annotations, out_file)
    print('all done!')


if __name__ == '__main__':
    # 1. convert dicom to numpy .npy file [train test split]
    # for split in ['test']:
    #     src_path = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/vinbigdata_chestXray/{}'.format(split)
    #     out_path = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/npy_{}'.format(split)
    #     json_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/vinbig_{}.json'.format(split)
    #     dicom_to_numpy_array(src_path, out_path, json_file, multiprocessing=True)

    # center_csv_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/data_center'
    # save_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/data_center/concate_all.csv'
    # concact_all_center_csvs(center_csv_dir, save_file)


    # print('testing reading speed.')
    # import time
    # def read_img(file):
    #     img = np.load(file)
    #     img = np.stack([img, img, img]).transpose((1, 2, 0))
    #     return img
    # file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/npy_train/051132a778e61a86eb147c7c6f564dfe.npy'
    # test_num = 200
    # start_time = time.time()
    # for _ in range(test_num):
    #     read_img(file)
    # end_time = time.time()
    # print('avg_time {} of {} test'.format((end_time - start_time) / test_num, test_num))

    # 3. getting the mean and std statistics of the dataset.
    # center_csv_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/data_center/concate_all.csv'
    # save_file = osp.join(osp.dirname(center_csv_file), 'concate_all_withMeanStd.csv')
    # get_ceneter_mean_std_mulpro(src_path, center_csv_file, save_file, num_center=13)


    # 4. adding mean and std statistics into the original dataset.
    for split in ['train', 'val']:
        src_anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/' \
                        'processed_data/new_Kfold_annotations/rad_id8/vinbig_{}_fold3.json'.format(split)
        out_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/' \
                   'processed_data/new_Kfold_annotations_MeanStd/rad_id8/vinbig_{}_fold3.json'.format(split)
        csv_info_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/data_center/concate_all_withMeanStd.csv'
        add_mean_std_to_jsonAnno(src_anno_file, out_file, csv_info_file)

