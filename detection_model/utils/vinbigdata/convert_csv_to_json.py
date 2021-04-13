import pandas as pd
import mmcv
import os
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, './')
from utils.vinbigdata import read_csv_files, load_annotations


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


def make_pesudo_labels(test_pred_csv, json_anno_file, shape_info_file, out_json_file, conf_thr=0.3, cls_map=None):
    """make pseudo label for the test set, and then merge with labeled train data.
    Args:
        test_pred_csv (str): path to load the prediction of test set.
        json_anno_file (str): path to load labeled training set.
        shape_info_file (str): path to load the img_info of the test images, e.g., width and height
        out_json_file (str): path to save merged labeled and pseudo label data.
        conf_thr (float): confidence threshold to filter the prediction boxes.
    """
    mmcv.mkdir_or_exist(os.path.dirname(out_json_file))
    test_pred_boxes = read_csv_files(test_pred_csv)[0]

    # load the shape of test images.
    shape_infos = mmcv.load(shape_info_file)
    shape_infos = {os.path.splitext(info['file_name'])[0]: info
                   for info in shape_infos['images']}

    # the annotations that we want to add pseudo label on
    all_annotations = mmcv.load(json_anno_file)

    images = all_annotations['images']
    annotations = all_annotations['annotations']
    # update the filename of train images
    for idx in range(len(images)):
        img_info = images[idx]
        file_name = img_info['file_name']
        img_info['file_name'] = 'train/{}'.format(file_name)
        images[idx] = img_info

    before_num =len(images)
    img_id_offset = 10000000
    anno_id_offset = 10000000
    left_imgs = 0
    num_left_boxes = []
    # filter the test prediction with lower confidence
    for idx, (im_id, pred_boxes) in tqdm(enumerate(test_pred_boxes.items()), total=len(test_pred_boxes)):
        keep_idxes = np.where((pred_boxes[:, 4] >= conf_thr) &
                              (pred_boxes[:, -1] != 14))[0]
        keep_boxes = pred_boxes[keep_idxes]
        if keep_boxes.shape[0] < 1:  # no boxes left after filtering
            continue
        left_imgs += 1
        num_left_boxes.append(keep_boxes.shape[0])
        im_info = shape_infos[im_id]
        image = dict(
            id=img_id_offset + idx,
            width=im_info['width'],
            height=im_info['height'],
            file_name='test/{}'.format(im_info['file_name']),
            date_captured=im_info['date_captured']
        )
        images.append(image)
        for anno_idx in range(keep_boxes.shape[0]):  # loop through boxes in one image
            anno_id_offset += 1
            annotation = dict(
                id=anno_id_offset,
                image_id=img_id_offset + idx,
                category_id=int(keep_boxes[anno_idx, -1]) + 1 if cls_map is None else cls_map[int(keep_boxes[anno_idx, -1])],
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
    mmcv.dump(all_annotations, out_json_file)
    print('all done!')


def filter_Nofinding_imgs(ori_ann_file, filter_info_file, out_file,
                          score_thr=0.08, key_name='class'):
    """Filter the images that have no lesions according to the classification model,
    in test set.
    Args:
        ori_ann_file (str): path to load json annotations
        filter_info_file (str): path to load csv filter infos.
        score_thr (float): score threshold to filter Nofinding images.
    """
    ori_ann_infos = mmcv.load(ori_ann_file)
    df = pd.read_csv(filter_info_file)

    ori_image_infos = {os.path.splitext(info['file_name'])[0]: info
                       for info in ori_ann_infos['images']}
    print('before filter, there are {} images.'.format(len(ori_image_infos)))
    new_images = []
    for idx, row in df.iterrows():
        image_name = row['image_id']
        cls = row[key_name]
        if cls >= score_thr:
            new_images.append(ori_image_infos[image_name])
    print('after filter, there are {} images.'.format(len(new_images)))
    print('saving new test annotations into file')
    ori_ann_infos['images'] = new_images
    mmcv.dump(ori_ann_infos, out_file)
    print('all done!')


def diff_test_imgs(test_info_1088_file, test_info_ern_file):
    test_info_1088 = mmcv.load(test_info_1088_file)
    test_info_ern = mmcv.load(test_info_ern_file)

    test_info_1088_imgs = set([info['id'] for info in test_info_1088['images']])
    test_info_ern_imgs = set([info['id'] for info in test_info_ern['images']])
    print('(ern - 1088): {}\nlen'.format(test_info_ern_imgs - test_info_1088_imgs))
    print('(ern + 1088): {}'.format(len(test_info_ern_imgs.union(test_info_1088_imgs))))


def merge_json_anno_files(file1, file2, pre_fixs, out_file):
    """
    Merge two json annos file (test info file) that have no label.
    Args:
        pre_fixs (list or tupele): file_name prefix
    """
    anno_infos1 = mmcv.load(file1)
    anno_infos2 = mmcv.load(file2)

    ori_img_infos = [anno_infos1['images'], anno_infos2['images']]
    assert len(ori_img_infos) == len(pre_fixs)

    new_images = []
    new_img_ids = []
    img_offsets = [i * 1000000 for i in range(len(pre_fixs))]
    for idx, pre_fix in enumerate(pre_fixs):  # loop through merging files
        for im_i in range(len(ori_img_infos[idx])):  # loop through imgs in one file
            img_info = ori_img_infos[idx][im_i]
            img_info['id'] += img_offsets[idx]
            new_img_ids.append(img_info['id'])
            img_info['file_name'] = '{}/{}'.format(pre_fixs[idx], img_info['file_name'])
            new_images.append(img_info)

    print('before merging: num_images: {}  {}'.format(len(ori_img_infos[0]), len(ori_img_infos[1])))
    print('after merging, num_images: ', len(new_images))
    print('num unique img_ids: ', len(list(set(new_img_ids))))

    print('saving new annotations...')
    anno_infos1['images'] = new_images
    mmcv.dump(anno_infos1, out_file)
    print('all done!')


if __name__ == '__main__':
    src_anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_' \
                    'AbnormDetect/processed_data/vinbig_test.json'
    filter_info_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/efn_classifier_test.csv'
    # out_file = os.path.join(os.path.dirname(src_anno_file), 'vinbig_test_cls_ern.json')
    # filter_Nofinding_imgs(src_anno_file, filter_info_file, out_file,
    #                       score_thr=0.04, key_name='cls_score_mean')

    # test_info_1088_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/vinbig_test_cls.json'
    # test_info_ern_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/vinbig_test_cls_ern.json'
    # diff_test_imgs(test_info_1088_file, test_info_ern_file)

    # test_pred_csv = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference_aafma/' \
    #                 'exp_yolov4_p6_cat7Slice_fold_annotationsF0_0306/merge_test_OneSliceF01234_5sliceF0_7sF0_wbf.csv'
    # test_pred_csv = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference_aafma/' \
    #                 'exp_yolov4_p6_PSLcat5Slice_fold_annotationsF0_0312/merge_test_1SF01234_5SF0PSL_7SF0_wbf.csv'
    test_pred_csv = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference_aafma/' \
                    'exp_yolov4_p6_cat5Slice_fold_annotationsF4_0310/merge_test_highestScore.csv'
    json_anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/A_AFMA_Detection/processed/fold_annotations/instances_train_fold{}.json'
    shape_info_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/A_AFMA_Detection/processed/aafma_test.json'
    out_json_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/A_AFMA_Detection/processed/pseudo_labels/pseudo0.5_train_fold{}.json'
    for fold in range(1):
        print('getting the pseudo label of fold {}, train + test'.format(fold))
        make_pesudo_labels(test_pred_csv, json_anno_file.format(fold),
                           shape_info_file, out_json_file.format(fold),
                           conf_thr=0.5,
                           cls_map={0: 2, 1: 11})

    # merging two files, for training the pseudo label with [train + val + test]
    # file1 = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/vinbig_test_cls.json'
    # file2 = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/new_Kfold_annotations/all/vinbig_val_fold3.json'
    # pre_fixs = ['test', 'train']
    # out_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/pseudo_labels/vinbig_Test_ValRallF3.json'
    # merge_json_anno_files(file1, file2, pre_fixs, out_file)

