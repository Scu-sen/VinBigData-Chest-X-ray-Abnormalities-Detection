import cv2
import os
from numpy import random
import numpy as np
from tqdm import tqdm
import mmcv
from collections import defaultdict
import pandas as pd

import sys
sys.path.insert(0, './')
from utils.vinbigdata import read_csv_files
from utils.general import plot_one_box
from utils.vinbigdata.eval_vinbigdata import VinBigData_class_names


np.random.seed(40)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(VinBigData_class_names))]


def vis_pred_from_csv(pred_csv_file, img_root, save_dir, score_thr=0.05, show_num=10, show_cls=None,
                      show_img_ids=None):
    """
    visualize prediction results from csv file
    Args:
        pred_csv_file (str): file to load prediction results
        img_root (str): path to load images that are going to show
        out_root (str): directory that save the visualization images
        show_img_ids: (str): path to load the showing img_ids.
    """
    os.makedirs(save_dir, exist_ok=True)
    pred_results, _ = read_csv_files(pred_csv_file)
    if show_img_ids is not None:
        df = pd.read_csv(show_img_ids)
        img_ids = df['file'].tolist()
    else:
        img_ids = list(pred_results.keys())
    cur_num_show = 0
    for idx, im_id in enumerate(tqdm(img_ids)):
        pred_boxes = pred_results[im_id]
        try:
            img = cv2.imread(os.path.join(img_root, im_id + '.png'))
            assert img is not None, 'image {} not found!'.format(im_id)
        except FileNotFoundError:
            continue
        keep_idxes = np.where(pred_boxes[:, 4] > score_thr)[0]
        if keep_idxes.shape[0] < 1:
            continue
        cur_num_show += 1
        pred_boxes = pred_boxes[keep_idxes]
        for box_i in range(pred_boxes.shape[0]):
            box_info = pred_boxes[box_i]
            score = box_info[4]
            if score < score_thr:
                continue
            box = box_info[:4]
            cls = int(box_info[-1])
            if show_cls is not None and show_cls != cls:
                continue
            label = '{}_{:.1f}'.format(VinBigData_class_names[int(cls)], score * 100)
            plot_one_box(box, img, label=label, color=colors[int(cls)],
                         line_thickness=3)

        save_path = os.path.join(save_dir, im_id + '.jpg')
        cv2.imwrite(save_path, img)
        if cur_num_show >= show_num:
            break
    print('all done!')


def vis_annos_from_json(anno_file, img_root, save_dir, show_num=10):
    """
        visualize prediction results from csv file
        Args:
            anno_file (str): file to load json annotations
            img_root (str): path to load images that are going to show
            save_dir (str): directory that save the visualization images
        """
    os.makedirs(save_dir, exist_ok=True)
    annotations = mmcv.load(anno_file)
    image_bboxes = defaultdict(list)
    image_names = {info['id']: info['file_name'] for info in annotations['images']}
    for anno in annotations['annotations']:
        bbox = anno['bbox']
        if anno['category_id'] == 15:
            continue
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        cls_id = anno['category_id'] - 1
        image_bboxes[anno['image_id']].append(bbox + [cls_id])

    for idx, (im_id, gt_boxes) in enumerate(tqdm(image_bboxes.items())):
        if len(gt_boxes) < 1:
            continue
        img = cv2.imread(os.path.join(img_root, image_names[im_id]))
        assert img is not None, 'image {} not found!'.format(im_id)

        gt_boxes = np.array(gt_boxes, dtype=np.int)
        for box_i in range(gt_boxes.shape[0]):
            box_info = gt_boxes[box_i]
            box = box_info[:4]
            cls = int(box_info[-1])
            label = '%s' % (VinBigData_class_names[int(cls)])
            plot_one_box(box, img, label=label, color=colors[int(cls)], line_thickness=2)

        save_path = os.path.join(save_dir, '{}_{}.jpg'.format(os.path.splitext(image_names[im_id])[0], idx))
        cv2.imwrite(save_path, img)
        if idx >= show_num:
            break
    print('all done!')


if __name__ == '__main__':
    img_root = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/data/vinbigdata/images/test'
    # img_root = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/data/rsna/images/train'
    show_num = 30
    pred_csv_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/280FinalModels_LowerThre'
    for filename in os.listdir(pred_csv_dir):
        print('visiting {}'.format(filename))
        pred_csv_file = os.path.join(pred_csv_dir, filename)
        save_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/Final_280Vis/{}'.format(
            os.path.splitext(filename)[0])

        vis_pred_from_csv(pred_csv_file, img_root, save_dir,
                          show_num=show_num, score_thr=0.2)

    # pred_csv_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference/yolov4_p6_0127/results_test_cls.csv'
    # save_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/YOLOV4_val0.464Test0.263'
    # vis_pred_from_csv(pred_csv_file, img_root, save_dir,
    #                   show_num=show_num)

    # pred_csv_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/mmdetection/' \
    #                 'work_dirs/DHFRCNN_rx101_64x4d_pafpn_BNmstrain_F1_0119/results/results_test_cls.csv'
    # save_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/DHRCNN_val0.400Test0.218'
    # vis_pred_from_csv(pred_csv_file, img_root, save_dir,
    #                   show_num=show_num)

    # pred_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference'
    # pred_csv_files = [os.path.join(pred_dir, 'exp_yolov4_p6_rad_id8F3_0130/results_test_cls.csv'),
    #                   os.path.join(pred_dir, 'exp_yolov4_p6_rad_id9F4_0128/results_test_cls.csv'),
    #                   os.path.join(pred_dir, 'exp_yolov4_p6_rad_id10F1_0129/results_test_cls.csv')]
    # save_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/vis_testPred_radI{}F{}'
    # rad_ids = (8, 9, 10)
    # folds = (3, 4, 1)
    # for pred_file, radId, fold in zip(pred_csv_files, rad_ids, folds):
    #     print('visualizing predictions of rad{} fold{}'.format(radId, fold))
    #     vis_pred_from_csv(pred_file,
    #                       img_root,
    #                       save_dir.format(radId, fold),
    #                       score_thr=0.1,
    #                       show_num=show_num)

    # pred_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/RSNA_PseudoLabels/merged_RSNA_pseudoLabels.csv'
    # vis_pred_from_csv(pred_file,
    #                   img_root,
    #                   save_dir='/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/RSNA_PseVis',
    #                   score_thr=0.4,
    #                   show_num=show_num,
    #                   show_cls=None)
    # center_root = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/test_center'
    # for center_file in ['test_no_spacing_pixl_1.csv', 'test_0.139000.csv']:  # ['test_0.140.csv', 'test_0.127.csv', 'test_no_spacing_no_pixl.csv']:
    #     print('visualizing center: ', center_file)
    #     vis_pred_from_csv(pred_file,
    #                       img_root,
    #                       save_dir='/mnt/group-ai-medical-2/private/zehuigong/torch_code/'
    #                                'ScaledYOLOv4/demo/testCenterVis/{}'.format(center_file[:-4]),
    #                       score_thr=0.2,
    #                       show_num=show_num,
    #                       show_cls=None,
    #                       show_img_ids=os.path.join(center_root, center_file))

    # anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/' \
    #             'processed_data/new_Kfold_annotations/rad_id{}/vinbig_val_fold{}.json'
    # img_root = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/train'
    # save_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/vis_radI{}F{}'
    # #
    # #
    # # for radId, fold in zip(rad_ids, folds):
    # #     print('visualizing annotations of rad{} fold{}'.format(radId, fold))
    # #     vis_annos_from_json(anno_file.format(radId, fold), img_root,
    # #                         save_dir.format(radId, fold), show_num=900)
    # #
    # vis_annos_from_json(anno_file, img_root, save_dir, show_num=10)
