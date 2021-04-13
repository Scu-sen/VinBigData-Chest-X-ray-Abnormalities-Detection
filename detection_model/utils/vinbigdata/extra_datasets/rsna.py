import numpy as np
import pydicom
import pandas as pd
import cv2
import os
import glob
from tqdm import tqdm
import mmcv
import time
from collections import defaultdict
import math
from multiprocessing import Pool, cpu_count

import sys
sys.path.insert(0, './')
from utils.vinbigdata.extra_datasets.SiimACR import info, licenses, bbox2mask
from utils.vinbigdata import read_csv_files


categories = [{"supercategory": "lesion", "id": 1, "name": "Pneumonia"}]


def read_csv(csv_file):
    """
    Read_csv file
    Args:
        csv_file:
    Returns:
    Dict{
        'image_id': list[list] [x1, y1, x2, y2, cls]
    }
    """
    results = defaultdict(list)
    df = pd.read_csv(csv_file)
    fields = ['x', 'y', 'width', 'height', 'Target']
    for idx, row in df.iterrows():
        patId = row['patientId']
        if math.isnan(row['x']):
            box = [0, 0, 1, 1, 15]
        else:
            box = [float(row[f]) for f in fields]
            box[2] += box[0]
            box[3] += box[1]

        results[patId].append(box)

    return results


def cvt_image_format(src_path, out_path):
    ds = pydicom.read_file(src_path)
    img = ds.pixel_array
    assert img.max() < 256, 'max value: {}'.format(img.max())
    mmcv.imwrite(img, out_path)
    return os.path.basename(src_path), img.shape


def get_image_annos_RSNA(data_root, save_dir, csv_anno_file=None, class_info_file=None,
                         multiprocessing=False, split='train'):
    """
    Read dicom data, and save it into png image.
    get bounding boxes from mask annotations
    1. Read csv annotations
    2. For each train dicom file, convert to png (optionally).
    3. get the bounding boxes annotations in coco format.
    4. save coco format annotations into json file
    """
    save_img_dir = save_dir + '/{}'.format(split)
    save_anno_dir = save_dir + '/annotations'
    mmcv.mkdir_or_exist(save_img_dir)
    mmcv.mkdir_or_exist(save_anno_dir)

    coco_annotations = {}
    images = []

    if split in ['train', 'val']:  # getting annotations for tain/val
        assert csv_anno_file is not None
        assert class_info_file is not None
        ori_annotations = read_csv(csv_anno_file)
        df_cls_info = pd.read_csv(class_info_file)
        patIds = sorted(list(ori_annotations.keys()))
    else:
        patIds = sorted([os.path.splitext(f)[0] for f in os.listdir(data_root)])

    print('start converting images...')
    img_shape_list = []
    if not os.path.exists(os.path.join(save_img_dir, 'img_shape_info.pkl')):
        if multiprocessing:
            pools = Pool(cpu_count())
            async_results = []
            for pid in patIds:
                async_results.append(
                    pools.apply_async(cvt_image_format,
                                      (os.path.join(data_root, pid + '.dcm'),
                                       os.path.join(save_img_dir, '{}.png'.format(pid)))))
            for async_res in async_results:
                async_res.wait()
                res = async_res.get()
                img_shape_list.append(res)
        else:
            for pid in patIds:
                img_shape_list.append(cvt_image_format(
                    os.path.join(data_root, pid + '.dcm'),
                     os.path.join(save_img_dir, '{}.png'.format(pid))
                ))
        mmcv.dump(img_shape_list, os.path.join(save_img_dir, 'img_shape_info.pkl'))
    else:
        img_shape_list = mmcv.load(os.path.join(save_img_dir, 'img_shape_info.pkl'))

    print('start getting annotations...')
    num_imgs = 0
    num_anno = 0
    annotations = []
    for idx, (patId, shape) in tqdm(enumerate(img_shape_list), total=len(img_shape_list)):
        patId = os.path.splitext(patId)[0]
        num_imgs += 1
        h, w = shape[:2]
        if split in ['train', 'val']:
            is_normal = df_cls_info[df_cls_info['patientId'] == patId]['class'].values[0] == 'Normal'
        else:
            is_normal = False

        image_info = dict(
            id=num_imgs,
            width=w,
            height=h,
            file_name='{}.png'.format(patId),
            date_captured=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            is_normal=is_normal,
        )
        images.append(image_info)

        # get bounding box annotations
        if split in ['train', 'val']:
            for box in ori_annotations[patId]:
                num_anno += 1
                x1, y1, x2, y2, cls = box
                segm, area = bbox2mask([x1, y1, x2, y2])
                anno_info = dict(
                    id=num_anno,
                    image_id=num_imgs,
                    category_id=cls,
                    iscrowd=0,
                    segmentation=segm,
                    area=area,
                    bbox=[x1, y1, x2 - x1, y2 - y1])
                annotations.append(anno_info)

    coco_annotations['annotations'] = annotations
    coco_annotations['info'] = info
    coco_annotations['licenses'] = licenses
    coco_annotations['categories'] = categories
    coco_annotations['images'] = images

    print('convert done\nsaving coco annotations')
    mmcv.dump(coco_annotations, os.path.join(save_anno_dir, 'rsna_{}_annos.json'.format(split)))
    print('all done!')


def get_json_file_forPseudoLabel(rsna_train_json, rsna_test_json, pre_fixs, out_file):
    """
    Get all the images on RSNA for detector's pseudo labelling.
    Args:
        rsna_train_json (str): path to load rsna training annotations file.
        rsna_test_json (str): path to load rsna test annotations file.
        pre_fixs (list[str]): file_name pre_fix to get the correct path of the images
                              e.g., rsna/images/train,   rsna/images/test
    """
    from utils.vinbigdata.vinbig_label import categories

    train_annos = mmcv.load(rsna_train_json)
    test_annos = mmcv.load(rsna_test_json)

    print('start getting new annotations')
    images = []
    num_images = 0
    for img_info in tqdm(train_annos['images']):  # loop through train images
        if img_info['is_normal']:  # skip with the normal images
            continue
        image = dict(
            id=num_images,
            width=img_info['width'],
            height=img_info['height'],
            file_name='{}/{}'.format(pre_fixs[0], img_info['file_name']),
        )
        images.append(image)
        num_images += 1

    num_images += 1
    for img_info in tqdm(test_annos['images']):  # loop through testing images
        image = dict(
            id=num_images,
            width=img_info['width'],
            height=img_info['height'],
            file_name='{}/{}'.format(pre_fixs[1], img_info['file_name']),
        )
        images.append(image)
        num_images += 1

    print('after merging, there are total of {} images'.format(len(images)))
    train_annos['images'] = images
    train_annos['annotations'] = []
    train_annos['categories'] = categories

    print('getting done! saving new annotations into file...')
    mmcv.dump(train_annos, out_file)
    print('all done!')


def xyxy2xywh(box):
    box[2:4] = box[2:4] - box[0:2]
    return list(map(int, box.tolist()))


def make_psedolabels_rsna(ori_anno_file, pseudo_pred_file, pseudo_info_files, out_file, conf_thr=0.3, cls_map=None):
    """
    add RSNA pseudo labels into the original json annotation file.
    Args:
        pseudo_pred_file (str): path to load the prediction of pseudo labels.
        ori_anno_file (str): path to load labeled training set, json file.
        pseudo_info_files (str or list[str]): path to load the img_info of the test images, e.g., width and height
        out_file (str): path to save merged labeled and pseudo label data.
        pre_fixs (list[str]): prefix to the file_name
        conf_thr (float): confidence threshold to filter the prediction boxes.
    """
    mmcv.mkdir_or_exist(os.path.dirname(out_file))

    print('reading pseudo predictions...')
    test_pred_boxes = read_csv_files(pseudo_pred_file)[0]

    # load the shape of test images.
    pseudo_img_infos = mmcv.load(pseudo_info_files)

    pseudo_img_infos = {os.path.splitext(os.path.basename(info['file_name']))[0]: info
                        for info in pseudo_img_infos['images']}

    # the annotations that we want to add pseudo label on
    all_annotations = mmcv.load(ori_anno_file)

    images = all_annotations['images']
    annotations = all_annotations['annotations']
    # update the filename of train images
    for idx in range(len(images)):
        img_info = images[idx]
        file_name = img_info['file_name']
        # vinbigdata/images/train/xxx.png
        img_info['file_name'] = 'vinbigdata/images/train/{}'.format(file_name)
        images[idx] = img_info

    before_num = len(images)
    img_id_offset = 10000000
    anno_id_offset = 10000000
    left_imgs = 0
    num_left_boxes = []
    # filter the pseudo prediction with lower confidence
    for idx, (im_id, pred_boxes) in tqdm(enumerate(test_pred_boxes.items()), total=len(test_pred_boxes)):
        keep_idxes = np.where((pred_boxes[:, 4] >= conf_thr) &
                              (pred_boxes[:, -1] != 14))[0]
        keep_boxes = pred_boxes[keep_idxes]
        if keep_boxes.shape[0] < 1:  # no boxes left after filtering
            continue
        left_imgs += 1
        num_left_boxes.append(keep_boxes.shape[0])
        im_info = pseudo_img_infos[im_id]
        image = dict(
            id=img_id_offset + idx,
            width=im_info['width'],
            height=im_info['height'],
            file_name=im_info['file_name'],
            # date_captured=im_info['date_captured']
        )
        images.append(image)
        for anno_idx in range(keep_boxes.shape[0]):  # loop through boxes in one image
            anno_id_offset += 1
            annotation = dict(
                id=anno_id_offset,
                image_id=img_id_offset + idx,
                category_id=int(keep_boxes[anno_idx, -1]) + 1 if cls_map is None else cls_map[
                    int(keep_boxes[anno_idx, -1])],
                iscrowd=0,
                bbox=xyxy2xywh(keep_boxes[anno_idx, :4].astype(np.int))
            )
            annotation['segmentation'], annotation['area'] = \
                bbox2mask(keep_boxes[anno_idx, :4])
            assert annotation['area'] > 1
            annotations.append(annotation)
    num_left_boxes = np.array(num_left_boxes)
    print('after filtering, there are {} test images left\nthe box info: mean:{:.2f}  min:{:.2f}  max:{:.2f}'.format(
        left_imgs, num_left_boxes.mean(), num_left_boxes.min(), num_left_boxes.max()))

    print('after_merging, there are {} images, before {} images'.format(len(images), before_num))

    all_annotations['images'] = images
    all_annotations['annotations'] = annotations

    print('convert done\nsaving coco annotations')
    mmcv.dump(all_annotations, out_file)
    print('all done!')



if __name__ == '__main__':
    # 1. getting the images and annotations for training set
    root_dir = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/RSNA'
    csv_anno_file = os.path.join(root_dir, 'stage_2_train_labels.csv')
    class_info_file = os.path.join(root_dir, 'stage_2_detailed_class_info.csv')
    # data_root = os.path.join(root_dir, 'stage_2_train_images')

    save_dir = os.path.join(root_dir, 'processed_data')
    # get_image_annos_RSNA(data_root, save_dir,
    #                      csv_anno_file=csv_anno_file,
    #                      class_info_file=class_info_file,
    #                      multiprocessing=True)

    # 2. getting the images and annotations for test set
    # data_root = os.path.join(root_dir, 'stage_2_test_images')
    # get_image_annos_RSNA(data_root, save_dir,
    #                      split='test',
    #                      multiprocessing=True)

    # 3. getting the images (both train and test) for pseudo labelling
    # rsna_train_json = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/RSNA/processed_data/annotations/rsna_train_annos.json'
    # rsna_test_json = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/RSNA/processed_data/annotations/rsna_test_annos.json'
    # pre_fixs = ['rsna/images/train', 'rsna/images/test']
    # out_file = os.path.join(save_dir, 'annotations', 'rsna_images_for_PSL.json')
    # get_json_file_forPseudoLabel(rsna_train_json, rsna_test_json, pre_fixs, out_file)

    # 4. get pseudo json annotations file
    rad_id = 'rad_id8'
    fold = 3
    ori_anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/' \
                    'processed_data/new_Kfold_annotations/{}/vinbig_train_fold{}.json'.format(rad_id, fold)
    pseudo_pred_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/RSNA_PseudoLabels/results_clsbest60_99RSNA_R8F3.csv'
    pseudo_info_files = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/RSNA/processed_data/annotations/rsna_images_for_PSL.json'
    out_json_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/RSNA/processed_data/' \
                    'pseudo_labels/{}/vinbig_train_fold{}.json'.format(rad_id, fold)
    # for fold in range(1):
    #     print('getting the pseudo label of fold {}, train + test'.format(fold))
    make_psedolabels_rsna(
        ori_anno_file,
        pseudo_pred_file,
        pseudo_info_files,
        out_json_file.format(fold),
        conf_thr=0.45)
