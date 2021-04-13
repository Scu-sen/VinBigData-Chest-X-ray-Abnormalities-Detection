"""
Date:   2021-01-21
Author: Zehui Gong
convert label format of vinbig data to txt
"""

import os
import numpy as np
import mmcv
from collections import defaultdict
import glob
import pandas as pd
import time
import copy
from tqdm import tqdm
import random
import csv

import sys
sys.path.insert(0, './')
from utils.vinbigdata import read_csv_files, bbox2csvline


categories = [{"supercategory": "lesion", "id": 1, "name": "Aortic enlargement"},
              {"supercategory": "lesion", "id": 2, "name": "Atelectasis"},
              {"supercategory": "lesion", "id": 3, "name": "Calcification"},
              {"supercategory": "lesion", "id": 4, "name": "Cardiomegaly"},
              {"supercategory": "lesion", "id": 5, "name": "Consolidation"},
              {"supercategory": "lesion", "id": 6, "name": "ILD"},
              {"supercategory": "lesion", "id": 7, "name": "Infiltration"},
              {"supercategory": "lesion", "id": 8, "name": "Lung Opacity"},
              {"supercategory": "lesion", "id": 9, "name": "Nodule/Mass"},
              {"supercategory": "lesion", "id": 10, "name": "Other lesion"},
              {"supercategory": "lesion", "id": 11, "name": "Pleural effusion"},
              {"supercategory": "lesion", "id": 12, "name": "Pleural thickening"},
              {"supercategory": "lesion", "id": 13, "name": "Pneumothorax"},
              {"supercategory": "lesion", "id": 14, "name": "Pulmonary fibrosis"},
              {"supercategory": "lesion", "id": 15, "name": "No finding"}]


def convert_annotations_one_file(src_file, out_label_dir, out_img_file=None):
    mmcv.mkdir_or_exist(out_label_dir)
    ori_annotations = mmcv.load(src_file)
    img_infos = {info['id']: info
                  for info in ori_annotations['images']}

    img_annotations = defaultdict(list)
    for anno in ori_annotations['annotations']:
        img_id = anno['image_id']
        cat_id = anno['category_id']
        if cat_id == 15:
            continue
        x, y, w, h = anno['bbox']
        ct_x = x + w * 0.5
        ct_y = y + h * 0.5
        # 1-based to 0-based
        img_annotations[img_id].append([
            cat_id - 1, ct_x, ct_y, w, h])

    # start convert
    for img_id, img_info in img_infos.items():  # loop through images
        img_h, img_w = img_info['height'], img_info['width']
        filename = os.path.splitext(img_info['file_name'])[0]
        img_label_file = os.path.join(out_label_dir, filename + '.txt')
        # one label file per image
        with open(img_label_file, 'w') as f:
            for obj in img_annotations[img_id]:
                cat_id, ct_x, ct_y, bw, bh = obj
                ct_x = ct_x / img_w
                ct_y = ct_y / img_h
                bw = bw / img_w
                bh = bh / img_h
                write_str = '{} {} {} {} {}\n'.format(cat_id, ct_x, ct_y, bw, bh)
                f.write(write_str)
    # write image infos
    if out_img_file is not None:
        with open(out_img_file, 'w') as f:
            for img_id in img_infos.keys():
                filename = img_infos[img_id]['file_name']
                f.write(filename + '\n')

    print('convert done!')

def convert_directory(src_dir, out_dir):
    """src_dir: directory to load K fold json annotation files
       structure of output label
       out_dir
       |--fold_K_out_label_dir
       |--|--img1_label.txt
       |__|--img2_label.txt
    """
    for anno_file in glob.glob(os.path.join(src_dir, '*.json')):
        # e.g. 'vinbig_val_fold5.json'
        fold_name = os.path.basename(anno_file)
        print('converting {} into txt annos'.format(fold_name))
        out_label_dir = os.path.join(out_dir, os.path.splitext(fold_name)[0])
        convert_annotations_one_file(anno_file, out_label_dir)


def Read_csv_annotation(csv_ann_file, include_radId=False):
    """
    Args:
        csv_ann_file (str): file to load annotations.
    Returns:
        Dict(): {
            'image_id': list[list]  [x1, y1, x2, y2, cls, [optional, rad_id]]
        }
    """
    annotations = defaultdict(list)
    clsid_to_clsname = dict()

    df = pd.read_csv(csv_ann_file)
    for idx, row in df.iterrows():
        image_id = row['image_id']
        cls_name = row['class_name']
        cls_id = row['class_id']
        rad_id = int(row['rad_id'][1:])
        x1, y1, x2, y2 = row['x_min'], row['y_min'], row['x_max'], row['y_max']

        # Normal slice, the coordinate is NAN
        if cls_name == 'No finding':
            x1, y1, x2, y2 = 0, 0, 1, 1

        clsid_to_clsname[cls_id] = cls_name
        box_info = [x1, y1, x2, y2, cls_id]
        if include_radId: box_info.append(rad_id)
        annotations[image_id].append(box_info)

    return annotations, clsid_to_clsname


def xyxy2xywh(box):
    box[2:4] = box[2:4] - box[0:2]
    return list(map(int, box.tolist()))


def bbox2mask(bbox):
    area = int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    polys = [bbox[0], bbox[1],  # tl
             bbox[2], bbox[1],
             bbox[2], bbox[3],
             bbox[0], bbox[3]
            ]
    polys = list(map(int, polys))
    return [polys], area


def get_AnnosRadWise(csv_ann_file, out_path, pre_coco_anno_file):
    """
    Getting the separate annotations for each radiologist.
    """
    os.makedirs(out_path, exist_ok=True)

    print('1. reading csv annotations...')
    csv_annotations, clsid_to_clsname = Read_csv_annotation(csv_ann_file, include_radId=True)

    pre_coco_ann_infos = mmcv.load(pre_coco_anno_file)
    img_shape_dict = {img_info['file_name']: (img_info['height'], img_info['width'])
                      for img_info in pre_coco_ann_infos['images']}

    coco_annotations = {}
    image = {"id": 0,
             'width': 0,
             'height': 0,
             'file_name': '',
             'license': 1,
             'flickr_url': '',
             'coco_url': '',
             'date_captured': ''}
    annotation = {'id': 0,
                  'image_id': 0,
                  'category_id': 0,
                  'segmentation': [],
                  'area': 0,
                  'bbox': [],
                  'iscrowd': 0}
    images = []
    annotations = []

    num_anno = 0
    num_img = 0
    for img_name, img_boxes in tqdm(csv_annotations.items()):  # loop through images
        img_boxes = np.array(img_boxes, dtype=np.int)  # (x1, y1, x2, y2, cls, radId)
        # filter Nofinding images
        keep_idxes = np.where(img_boxes[:, -2] != 15)[0]
        if keep_idxes.shape[0] < 1:
            continue
        img_boxes = img_boxes[keep_idxes]
        annoted_rads = np.unique(img_boxes[:, -1])

        for radId in annoted_rads:  # loop through radIds, for each radId, create one annotations
            keep_idxes = np.where(img_boxes[:, -1] == radId)[0]
            rad_img_boxes = img_boxes[keep_idxes]
            h, w = img_shape_dict[img_name + '.png']

            image['id'] = num_img + 1
            image['width'] = w
            image['height'] = h
            image['file_name'] = img_name + '.png'
            image['rad_id'] = int(radId)
            time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            image['date_captured'] = time_now
            images.append(copy.deepcopy(image))

            for anno_idx in range(rad_img_boxes.shape[0]):  # loop through objs in one image
                num_anno += 1
                annotation['id'] = num_anno
                annotation['image_id'] = num_img + 1
                annotation['category_id'] = int(rad_img_boxes[anno_idx, 4]) + 1
                annotation['iscrowd'] = 0
                annotation['segmentation'], annotation['area'] = \
                    bbox2mask(rad_img_boxes[anno_idx, :4])
                annotation['bbox'] = xyxy2xywh(rad_img_boxes[anno_idx, :4])
                annotations.append(copy.deepcopy(annotation))
                annotation['segmentation'].clear()

            num_img += 1

    coco_annotations['info'] = {}
    coco_annotations['licenses'] = {}
    coco_annotations['categories'] = categories
    coco_annotations['annotations'] = annotations
    coco_annotations['images'] = images

    print('saving coco annotations...')
    mmcv.dump(coco_annotations, os.path.join(out_path, 'vinbig_duplicateImgs.json'))
    print('all done!')


def split_normal_imgs(csv_anno_file, json_anno_file, out_file, normal_val_num=2500):
    """
    Split Normal images into two group, one for training, one for testing
    out_csv:
    image_id       fold                height        width
    50dsadas        0 (validation)      1500          1532
    50dsghds        1 (training)
    """
    random.seed(40)

    df = pd.read_csv(csv_anno_file)
    Normal_img_ids = df[df['class_name'] == 'No finding']['image_id'].unique().tolist()

    print('there are {} normal images'.format(len(Normal_img_ids)))

    random.shuffle(Normal_img_ids)
    val_normal_imgIds = Normal_img_ids[:normal_val_num]
    train_norml_imgIds = Normal_img_ids[normal_val_num:]

    # get img_shaoe infos
    json_annos = mmcv.load(json_anno_file)
    img_shapes = {os.path.splitext(info['file_name'])[0]: (info['height'], info['width'])
                  for info in json_annos['images']}

    df_save = pd.DataFrame()
    Normal_img_ids = val_normal_imgIds + train_norml_imgIds
    df_save['image_id'] = Normal_img_ids
    df_save['fold'] = [0 if i < normal_val_num else 1
                       for i in range(len(Normal_img_ids))]
    df_save['height'] = [img_shapes[im_id][0] for im_id in Normal_img_ids]
    df_save['width'] = [img_shapes[im_id][1] for im_id in Normal_img_ids]
    df_save.to_csv(out_file, index=False)


def add_normal_imgs_to_jsonAnno(src_dir, out_dir, normal_imgs_info_file):
    """
    Adding normal images to original json coco format annotations.
    Args:
        src_dir (str): directory to load coco annotations
        out_dir (str): directory to save updated annotations
        normal_imgs_info_file (str): path to load normal_imgs img_ids
    """
    def add_normal_imgIds(src_file, out_file, df):
        src_json_annos = mmcv.load(src_file)
        images = src_json_annos['images']
        annotations = src_json_annos['annotations']

        img_offset = 100000
        num_anno = 500000
        for idx, row in df.iterrows():  # loop through normal images
            im_id = row['image_id']

            image = dict(
                id=idx + img_offset,
                width=row['width'],
                height=row['height'],
                file_name=im_id + '.png')
            images.append(image)

            anno = dict(
                id=num_anno,
                image_id=idx + img_offset,
                category_id=15,
                iscrowd=0,
                segmentation=[[0, 0, 1, 0, 1, 1, 0, 1]],
                area=10,
                bbox=[0, 0, 1, 1])
            annotations.append(anno)
            num_anno += 1
        src_json_annos['images'] = images
        src_json_annos['annotations'] = annotations
        mmcv.dump(src_json_annos, out_file)

    # print('getting training and validation files...')
    src_train_files = glob.glob(os.path.join(src_dir, 'vinbig_train_*.json'))
    src_val_files = glob.glob(os.path.join(src_dir, 'vinbig_val_*.json'))

    # print('getting normal imgs infos...')
    df_normal = pd.read_csv(normal_imgs_info_file)
    df_val_normal = df_normal[df_normal['fold'] == 0]
    df_train_normal = df_normal[df_normal['fold'] == 1]

    mmcv.mkdir_or_exist(out_dir)
    # print('adding normal img infos into original train json file')
    for train_file in src_train_files:
        print('processing {}'.format(os.path.basename(train_file)))
        out_file = os.path.join(out_dir, os.path.basename(train_file))
        add_normal_imgIds(train_file, out_file, df=df_train_normal)

    # print('adding normal img infos into original val json file')
    for val_file in src_val_files:
        print('processing {}'.format(os.path.basename(val_file)))
        out_file = os.path.join(out_dir, os.path.basename(val_file))
        add_normal_imgIds(val_file, out_file, df=df_val_normal)

    print('all done!')


def check_anno_corretness():
    anno_file_ori = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/' \
                    'new_Kfold_annotations/all/vinbig_train_fold3.json'
    anno_file_normal = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/' \
                       'processed_data/new_Kfold_annotations_WithNormal/all/vinbig_train_fold3.json'
    from collections import defaultdict
    annos_ori = mmcv.load(anno_file_ori)
    img_infos_ori = {info['id']: info for info in annos_ori['images']}

    annos_normal = mmcv.load(anno_file_normal)
    img_infos_normal = {info['id']: info for info in annos_normal['images']}

    img_boxes_ori = defaultdict(list)
    img_boxes_normal = defaultdict(list)
    for annos in annos_ori['annotations']:
        img_boxes_ori[annos['image_id']].append(annos['bbox'] + [annos['category_id']])
    for annos in annos_normal['annotations']:
        img_boxes_normal[annos['image_id']].append(annos['bbox'] + [annos['category_id']])

    # for im_id in list(img_infos_ori.keys()):
    #     ori_h, ori_w = img_infos_ori[im_id]['height'], img_infos_ori[im_id]['width']
    #     normal_h, normal_w = img_infos_normal[im_id]['height'], img_infos_normal[im_id]['width']
    #
    #     boxes_ori = np.array(img_boxes_ori[im_id], dtype=np.int)
    #     boxes_normal = np.array(img_boxes_normal[im_id], dtype=np.int)
    #
    #     box_diff = (boxes_ori - boxes_normal).sum()
    #     assert ori_h == normal_h and ori_w == normal_w and box_diff == 0, '{} different, ori:()  normal:{}'.format(
    #         img_infos_ori['file_name'], (ori_h, ori_w), (normal_h, normal_w))
    #
    #     if im_id == 10:
    #         print(boxes_ori, boxes_normal)

    img_labels = defaultdict(list)
    for ann_info in annos_normal['annotations']:
        im_id = ann_info['image_id']
        img_labels[im_id].append(ann_info['category_id'])
    normal_imgIds = []
    abnormal_imgIds = []
    for im_id, im_labels in img_labels.items():
        if len(im_labels) == 1 and im_labels[0] == 15:  # normal image (No finding)
            normal_imgIds.append(im_id)
        else:
            abnormal_imgIds.append(im_id)
    print('{} normal imgIds,\t{} abnormal imgIds'.format(len(normal_imgIds), len(abnormal_imgIds)))


def filter_low_score_boxes(test_pred_file, out_file, score_thr=0.2):
    # x1, y1, x2, y2, score, cls
    pred_img_boxes, _ = read_csv_files(test_pred_file)

    print('3. saving new predictions into out_file...')
    with open(out_file, 'w') as f_out:
        csv_weiter = csv.writer(f_out)
        csv_weiter.writerow(['image_id', 'PredictionString'])
        for im_name, im_boxes in pred_img_boxes.items():
            keep_idxes = np.where(im_boxes[:, 5] != 14)[0]
            im_boxes = im_boxes[keep_idxes]
            csv_line = bbox2csvline(im_boxes, im_name, pre_scale=1.0, score_thr=score_thr)
            csv_weiter.writerow(csv_line)
    print('all done!')


if __name__ == '__main__':
    # root_path = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data'
    # postfix = 'new_Kfold_annotations'
    # out_root_path = os.path.join(root_path, 'txt_annotations', postfix)
    # # convert new_k_fold annotations
    # for rad_id in os.listdir(root_path + '/' + postfix):
    #     src_dir = os.path.join(root_path, postfix, rad_id)
    #     out_dir = os.path.join(out_root_path, rad_id)
    #     convert_directory(src_dir, out_dir)

    # csv_ann_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/vinbigdata_chestXray/train.csv'
    # out_path = os.path.join(root_path, 'rad_annotations')
    # pre_coco_anno_file = os.path.join(root_path, 'vinbig_train.json')
    #
    # get_AnnosRadWise(csv_ann_file, out_path, pre_coco_anno_file)

    # out_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/' \
    #            'VinBigdata_AbnormDetect/vinbigdata_chestXray/normal_imgs_split.csv'
    # json_anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/vinbig_train.json'
    # split_normal_imgs(csv_ann_file, json_anno_file, out_file)


    root_dir = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/' \
               'processed_data/new_Kfold_annotations'
    out_root_dir = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/' \
                   'processed_data/new_Kfold_annotations_WithNormal'
    # for file in os.listdir(root_dir):
    #     add_normal_imgs_to_jsonAnno(
    #         src_dir=os.path.join(root_dir, file),
    #         out_dir=os.path.join(out_root_dir, file),
    #         normal_imgs_info_file=out_file)


    # check_anno_corretness()

    test_pred_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/0.280_nmsnolimit_aortic_cls10_multimul_cls14mul4cls35_multithres13_1024_1280.csv'
    out_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/0.280_nmsnolimit_aortic_cls10_multimul_cls14mul4cls35_multithres13_1024_1280_filterLowScore.csv'
    filter_low_score_boxes(test_pred_file, out_file, score_thr=0.2)





