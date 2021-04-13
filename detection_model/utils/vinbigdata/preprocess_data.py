"""
Author: Zehui Gong
Date: 2021/01/03
"""

import pandas as pd
import numpy as np
import cv2
import os
from collections import defaultdict
from tqdm import trange
import time
import copy
import mmcv
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from multiprocessing import Pool, cpu_count
import argparse


info = {"description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01"}
licenses = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
             "name": "Attribution-NonCommercial-ShareAlike License"},
            {"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 2,
             "name": "Attribution-NonCommercial License"},
            {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/", "id": 3,
             "name": "Attribution-NonCommercial-NoDerivs License"},
            {"url": "http://creativecommons.org/licenses/by/2.0/", "id": 4, "name": "Attribution License"},
            {"url": "http://creativecommons.org/licenses/by-sa/2.0/", "id": 5,
             "name": "Attribution-ShareAlike License"},
            {"url": "http://creativecommons.org/licenses/by-nd/2.0/", "id": 6,
             "name": "Attribution-NoDerivs License"},
            {"url": "http://flickr.com/commons/usage/", "id": 7, "name": "No known copyright restrictions"},
            {"url": "http://www.usa.gov/copyright.shtml", "id": 8, "name": "United States Government Work"}]
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


def dicom2array(path, voi_lut=True, fix_monochrome=True):
    """ Convert dicom file to numpy array

    Args:
        path (str): Path to the dicom file to be converted
        voi_lut (bool): Whether or not VOI LUT is available
        fix_monochrome (bool): Whether or not to apply monochrome fix

    Returns:
        Numpy array of the respective dicom file

    """
    # Use the pydicom library to read the dicom file
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # The XRAY may look inverted
    #   - If we want to fix this we can
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    # Normalize the image array and return
    data = data - np.min(data)
    data = data / np.max(data)
    data *= 255
    return data.astype(np.uint8)


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


def filter_overlapped_boxes(annotations, iou_thr=0.5):
    """
    For the training set, there are three radiologists to annotate one slice independently,
    therefore, there may be more than one ground-truth boxes for one lesion.
     we need to remove the overlaps using the mean coordinates of these overlapped boxes.
    Args:
        annotations (dict[list]):
        {
            'image_id': list[list]  [x1, y1, x2, y2, cls, [optional, rad_id]]
        }
        iou_thr (float):

    Returns:
        new_annotations (dict[np.ndarray]):
        {
            'image_id' (np.ndarray):  [x1, y1, x2, y2, cls]
        }
    """
    new_annotations = dict()
    for img_id, annos in annotations.items():  # loop through images
        annos = np.array(annos).astype(np.float32)
        unique_clses = np.unique(annos[:, 4])
        new_img_boxes = []
        for cls in unique_clses:  # loop through classes
            idxes = np.where(annos[:, 4] == cls)[0]
            cls_annos = annos[idxes]
            x1, x2 = cls_annos[:, 0], cls_annos[:, 2]
            y1, y2 = cls_annos[:, 1], cls_annos[:, 3]

            areas = (x2 - x1) * (y2 - y1)
            order = np.arange(idxes.shape[0])
            new_cls_boxes = []
            while order.size > 0:
                i = order[0]
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                # merge overlap boxes
                inds = np.where(ovr > iou_thr)[0]
                overlap_boxes = np.vstack([cls_annos[i:i+1, :],
                                           cls_annos[order[inds + 1], :]])
                new_cls_boxes.append(np.mean(overlap_boxes, axis=0))

                # update order
                inds = np.where(ovr <= iou_thr)[0]
                order = order[inds + 1]
            new_img_boxes.extend(new_cls_boxes)
        new_annotations[img_id] = np.array(new_img_boxes, dtype=np.float32)

    return new_annotations


def bbox2mask(bbox):
    area = int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    polys = [bbox[0], bbox[1],  # tl
             bbox[2], bbox[1],
             bbox[2], bbox[3],
             bbox[0], bbox[3]
            ]
    polys = list(map(int, polys))
    return [polys], area


def cvt_image_format(src_path, out_path):
    img = dicom2array(src_path)
    mmcv.imwrite(img, out_path)
    return os.path.basename(src_path), img.shape


def xyxy2xywh(box):
    box[2:4] = box[2:4] - box[0:2]
    return list(map(int, box.tolist()))


def convert_to_cocojson(src_path, out_path, iou_thr=0.5, multiprocessing=False):
    """
    Function: To convert the annotation format form csv file to COCO json annotation,
          and at the same time, merge multi-boxes for one lesion into one!
          1. Read the csv annotation merge annotation slice-wise,
          2. filter overlapped boxes using NMS;
          3. convert annotation into COCO json format;
          4. save new annotation into file
    Args:
        src_path (str):
        out_path (str):
    """
    csv_ann_file = os.path.join(src_path, 'train.csv')
    data_path = os.path.join(src_path, 'train')

    os.makedirs(out_path, exist_ok=True)
    ann_out_file = os.path.join(out_path, 'vinbig_train.json')
    png_data_out_path = os.path.join(out_path, 'train')

    print('reading csv annotations')
    csv_annotations, clsid_to_clsname = Read_csv_annotation(csv_ann_file)
    # vis_idxes = np.random.randint(0, len(csv_annotations), size=20).tolist()
    # visualize_one_image(data_path, csv_annotations,
    #                     idxes=vis_idxes,
    #                     clsid_to_clsname=clsid_to_clsname,
    #                     save_dir='/mnt/group-ai-medical-2/private/zehuigong/torch_code/mmdetection/demo/before_filter')
    print('filtering overlapping bounding boxes')
    csv_annotations = filter_overlapped_boxes(csv_annotations, iou_thr=iou_thr)
    # visualize_one_image(data_path, csv_annotations,
    #                     idxes=vis_idxes,
    #                     clsid_to_clsname=clsid_to_clsname,
    #                     save_dir='/mnt/group-ai-medical-2/private/zehuigong/torch_code/mmdetection/demo/after_filter')
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
    image_ids = list(csv_annotations.keys())
    print('start converting annotations')

    img_shape_list = []
    if multiprocessing:
        pools = Pool(cpu_count())
        async_results = []
        for img_id in image_ids:
            async_results.append(
                pools.apply_async(cvt_image_format,
                                  (os.path.join(data_path, img_id + '.dicom'),
                                   os.path.join(png_data_out_path, '{}.png'.format(img_id)))))
        for async_res in async_results:
            async_res.wait()
            res = async_res.get()
            img_shape_list.append(res)
    else:
        for img_id in image_ids:
            img_shape_list.append(cvt_image_format(
                os.path.join(data_path, img_id + '.dicom'),
                os.path.join(png_data_out_path, '{}.png'.format(img_id))
            ))

    for img_i in trange(len(img_shape_list)):  # loop through images
        img_id = os.path.splitext(img_shape_list[img_i][0])[0]
        h, w = img_shape_list[img_i][1][:2]

        # img = dicom2array(os.path.join(data_path, img_id + '.dicom'))
        # mmcv.imwrite(img, os.path.join(png_data_out_path, '{}.png'.format(img_id)))
        # h, w = img.shape[:2]

        image['id'] = img_i + 1
        image['width'] = w
        image['height'] = h
        image['file_name'] = img_id + '.png'
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        image['date_captured'] = time_now
        images.append(copy.deepcopy(image))

        for anno_idx in range(csv_annotations[img_id].shape[0]):  # loop through objs in one image
            num_anno += 1
            annotation['id'] = num_anno
            annotation['image_id'] = img_i + 1
            annotation['category_id'] = int(csv_annotations[img_id][anno_idx, -1]) + 1
            annotation['iscrowd'] = 0
            annotation['segmentation'], annotation['area'] = \
                bbox2mask(csv_annotations[img_id][anno_idx, :4])
            annotation['bbox'] = xyxy2xywh(csv_annotations[img_id][anno_idx, :4])
            annotations.append(copy.deepcopy(annotation))
            annotation['segmentation'].clear()

    coco_annotations['info'] = info
    coco_annotations['licenses'] = licenses
    coco_annotations['categories'] = categories
    coco_annotations['annotations'] = annotations
    coco_annotations['images'] = images

    print('convert done\nsaving coco annotations')
    mmcv.dump(coco_annotations, ann_out_file)
    print('all done!')


def split_image_bboxes(annotations, rad_id):
    independent_annotations = {}
    for img_id in list(annotations.keys()):
        img_bboxes = np.array(annotations[img_id], dtype=np.int)
        idxes = np.where(img_bboxes[:, -1] == rad_id)[0]
        if idxes.shape[0] < 1:
            continue
        rad_boxes = img_bboxes[idxes]
        independent_annotations[img_id] = rad_boxes[:, :-1]
    return independent_annotations


def convert_csv_to_cocoV2(src_path, out_path, rad_ids=[], pre_coco_anno_path=None, debug=False):
    """The idea of senYang
    Get the annotations of radiologists !!independtly!!, and train three independent model,
    finally, merge the results of these three models.
      1. Read the csv annotation
      2. Split the original annotations into three independent annotations.
      3. convert annotation into COCO json format;
      4. save new annotation into file
    Args:
        src_path:
        out_path:
        pre_coco_anno_path (str): path to load pre-coco annotations, to get image height and width.
        rad_ids (list): which rad_id we want to get the annotations.
    Returns:

    """
    assert isinstance(rad_ids, list)
    csv_ann_file = os.path.join(src_path, 'train.csv')
    data_path = os.path.join(src_path, 'train')
    os.makedirs(out_path, exist_ok=True)

    print('1. reading csv annotations...')
    csv_annotations, clsid_to_clsname = Read_csv_annotation(csv_ann_file, include_radId=True)

    pre_coco_ann_infos = mmcv.load(pre_coco_anno_path)
    img_shape_dict = {img_info['file_name']: (img_info['height'], img_info['width'])
                      for img_info in pre_coco_ann_infos['images']}

    for rad_id in rad_ids:
        print('2. Getting the annotations of Radiologist {}.'.format(rad_id))
        rad_annotations = split_image_bboxes(csv_annotations, rad_id)

        # if debug:
        #     print('visualize the annotations of rad_id: {}'.format(rad_id))
        #     vis_idxes = np.random.randint(0, len(rad_annotations), size=20).tolist()
        #     visualize_one_image(
        #         data_path,
        #         rad_annotations,
        #         idxes=vis_idxes,
        #         clsid_to_clsname=clsid_to_clsname,
        #         save_dir='/mnt/group-ai-medical-2/private/zehuigong/torch_code/mmdetection/demo/rad_id%d' % rad_id)

        print('3. Converting annotations into coco format.')
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
        for img_i, (img_name, img_boxes) in enumerate(rad_annotations.items()):
            h, w = img_shape_dict[img_name + '.png']

            image['id'] = img_i + 1
            image['width'] = w
            image['height'] = h
            image['file_name'] = img_name + '.png'
            time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            image['date_captured'] = time_now
            images.append(copy.deepcopy(image))

            for anno_idx in range(img_boxes.shape[0]):  # loop through objs in one image
                num_anno += 1
                annotation['id'] = num_anno
                annotation['image_id'] = img_i + 1
                annotation['category_id'] = int(img_boxes[anno_idx, -1]) + 1
                annotation['iscrowd'] = 0
                annotation['segmentation'], annotation['area'] = \
                    bbox2mask(img_boxes[anno_idx, :4])
                annotation['bbox'] = xyxy2xywh(img_boxes[anno_idx, :4])
                annotations.append(copy.deepcopy(annotation))
                annotation['segmentation'].clear()

        coco_annotations['info'] = info
        coco_annotations['licenses'] = licenses
        coco_annotations['categories'] = categories
        coco_annotations['annotations'] = annotations
        coco_annotations['images'] = images

        print('saving coco annotations of rad_id {}'.format(rad_id))
        mmcv.dump(coco_annotations, os.path.join(out_path, 'vinbig_trainR{}.json'.format(rad_id)))
        print('all done!')


def get_test_annoCoco(test_data_path, out_path, multiprocessing=False):
    ann_out_file = os.path.join(out_path, 'vinbig_test.json')
    png_out_path = os.path.join(out_path, 'test')

    img_names = os.listdir(test_data_path)
    coco_annotations = {}
    image = {"id": 0,
             'width': 0,
             'height': 0,
             'file_name': '',
             'license': 1,
             'flickr_url': '',
             'coco_url': '',
             'date_captured': ''}
    images = []

    img_shape_list = []
    if multiprocessing:
        pools = Pool(cpu_count())
        async_results = []
        for img_id in img_names:
            async_results.append(
                pools.apply_async(cvt_image_format,
                                  (os.path.join(test_data_path, img_id),
                                   os.path.join(png_out_path, '{}.png'.format(os.path.splitext(img_id)[0])))))
        for async_res in async_results:
            async_res.wait()
            res = async_res.get()
            img_shape_list.append(res)
    else:
        for img_id in img_names:
            img_shape_list.append(cvt_image_format(
                os.path.join(test_data_path, img_id),
                os.path.join(png_out_path, '{}.png'.format(os.path.splitext(img_id)[0]))
            ))

    for img_i in trange(len(img_shape_list)):
        # im = dicom2array(os.path.join(test_data_path, im_name))
        # mmcv.imwrite(im, os.path.join(png_out_path, '{}.png'.format(os.path.splitext(im_name)[0])))
        # h, w = im.shape[:2]

        im_name = os.path.splitext(img_shape_list[img_i][0])[0]
        h, w = img_shape_list[img_i][1][:2]

        image['id'] = img_i + 50000
        image['width'] = w
        image['height'] = h
        image['file_name'] = im_name + '.png'
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        image['date_captured'] = time_now
        images.append(copy.deepcopy(image))
    coco_annotations['info'] = info
    coco_annotations['licenses'] = licenses
    coco_annotations['categories'] = categories
    coco_annotations['annotations'] = []
    coco_annotations['images'] = images

    print('saving coco annotations')
    mmcv.dump(coco_annotations, ann_out_file)
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


def change_annotations(src_anno_file, dst_anno_file):
    annotations = mmcv.load(src_anno_file)
    for anno in annotations['annotations']:
        anno['segmentation'] = [anno['segmentation']]

    mmcv.dump(annotations, dst_anno_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='', help='path to source data directory')
    parser.add_argument('--out_dir', type=str, default='data/vinbigdata/images', help='path to source data directory')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # data_root = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect'
    # out_root = os.path.join(data_root, 'processed_data')
    # # 1. get training annotations
    # convert_to_cocojson(src_path=data_root,
    #                     out_path=out_root,
    #                     iou_thr=0.25,
    #                     multiprocessing=True)
    #
    # # 2. get testing annotations
    # test_data_path = os.path.join(data_root, 'vinbigdata_chestXray/test')
    # get_test_annoCoco(test_data_path, out_root, multiprocessing=True)

    # 4. convert annotation version 2
    # convert_csv_to_cocoV2(src_path=data_root,
    #                       out_path=os.path.join(out_root, 'rad_annotations'),
    #                       rad_ids=[8, 9, 10],
    #                       pre_coco_anno_path=os.path.join(out_root, 'vinbig_train.json'),
    #                       debug=False)

    args = parse_args()
    # 1. get training images and annotations
    convert_to_cocojson(src_path=args.src_dir,
                        out_path=args.out_dir,
                        iou_thr=0.25,
                        multiprocessing=True)

    # 2. get testing images and annotations
    test_data_path = os.path.join(args.src_dir, 'test')
    get_test_annoCoco(test_data_path, args.out_dir, multiprocessing=True)
