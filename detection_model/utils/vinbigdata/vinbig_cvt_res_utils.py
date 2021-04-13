import os
import os.path as osp
import numpy as np
import csv
import pandas as pd
import mmcv

import sys
sys.path.append('./')
from utils.vinbigdata.mean_ap import eval_map
from utils.vinbigdata.utils import read_csv_files, bbox2csvline, read_csv


def result2vinbigdata(results, dst_path, img_names, pre_scale=0.5, score_thr=0.05):
    # assert len(results) == len(
    #     img_names), 'length of results must equal with length of img_names'
    dst_dir = os.path.dirname(dst_path)
    if not osp.exists(dst_dir):
        os.mkdir(dst_dir)
    dst_path = os.path.splitext(dst_path)[0] + '.csv'
    with open(dst_path, 'w') as f_out:
        csv_weiter = csv.writer(f_out)
        csv_weiter.writerow(['image_id', 'PredictionString'])
        for img_id, result in enumerate(results):  # loop through images
            labels = np.hstack([np.full((bbox.shape[0], ), cls, dtype=np.float32)
                                for cls, bbox in enumerate(result)])
            bboxes = np.vstack(result)

            # (x1, y1, x2, y2, score, cls)
            bboxes = np.hstack([bboxes, labels[:, np.newaxis]])
            csv_line = bbox2csvline(bboxes, img_names[img_id], pre_scale, score_thr)
            csv_weiter.writerow(csv_line)
    return True


def results2vinbigdata_twoThr(src_csv_file, out_csv_file, high_thr=0.1, low_thr=0.001):
    """post-processing the detection results using two-threshold method.
       Firstly, using the high_thr to filter normal images, for the left images, using
       low_thr to keep as most predictions as possible.
    """
    ori_results = read_csv_files(src_csv_file)[0]
    with open(out_csv_file, 'w') as f_out:
        csv_weiter = csv.writer(f_out)
        csv_weiter.writerow(['image_id', 'PredictionString'])
        for im_id in ori_results.keys():  # loop through images that have detections
            ori_bboxes = ori_results[im_id]
            max_score= np.max(ori_bboxes[:, 4])
            if max_score < high_thr:  # normal image (No finding)
                csv_weiter.writerow([im_id, '14 1 0 0 1 1'])
            else:
                csv_line = bbox2csvline(ori_bboxes, im_id, 1.0, score_thr=low_thr)
                csv_weiter.writerow(csv_line)
    print('filter done!')


def two_class_filter(res_file, filter_file, sub_sample_file, out_file):
    res_det = read_csv(res_file)
    sample_sub = read_csv(sub_sample_file)
    df_filter = pd.read_csv(filter_file)

    remove_imgIds = df_filter[df_filter['target'] < 0.08]['image_id'].tolist()
    for reId in remove_imgIds:
        res_det[reId] = sample_sub[reId]

    with open(out_file, 'w') as fout:
        csv_weiter =csv.writer(fout)
        csv_weiter.writerow(['image_id', 'PredictionString'])
        for img_id, res_str in res_det.items():  # loop through images
            line = [img_id, res_str]
            csv_weiter.writerow(line)
    print('done!')


def eval_vinbigdata_voc(results, gt_annos, pre_scale=0.5, score_thr=0.05):
    """
    evaluate the results for vinbigdata.
    Args:
        results (list[list[np.ndarray]]): list of list, [[[x1, y1, x2, y2, score], cls_j, ...], [img_i], ...]
        which each element is prediction of image i, class j.
        gt_annos (list[np.ndarray]): the corresponding ground-truth boxes for images.
        pre_scale (float):
        score_thr (float):
    Returns:
    """
    # gt bboxes is in original image coordinate system.
    gt_bboxes = [anno[:, :4] * pre_scale for anno in gt_annos]
    gt_labels = [anno[:, 4] for anno in gt_annos]
    annotations = []
    for i in range(len(gt_bboxes)):
        anno = dict(
            bboxes=gt_bboxes[i],
            labels=gt_labels[i])
        annotations.append(anno)

    num_classes = len(results[0])
    # filter prediction boxes
    for img_i in range(len(results)):
        for cls in range(num_classes):
            bboxes = results[img_i][cls]
            idxes = np.where(bboxes[:, -1] >= score_thr)[0]
            results[img_i][cls] = bboxes[idxes]

    mean_ap, eval_results = eval_map(
        results,
        annotations=annotations,
        iou_thr=0.4)
    return mean_ap


def eval_vinbigdata_cls(results, gt_annos):
    """
    evaluate the results for classification performance in vinbigdata.
    ACC and Recall.
    Args:
        results (list[tuple[np.ndarray]]): [scores, pred_labels]
        gt_annos (list[np.array]): gt_label for each image
    Returns:
    """
    pred_labels = np.concatenate([res[1] for res in results],
                                 axis=0)
    gt_labels = np.concatenate(gt_annos, axis=0)
    assert pred_labels.shape[0] == gt_labels.shape[0]

    acc = (pred_labels == gt_labels).sum() / gt_labels.shape[0] * 100.

    tp = np.where((gt_labels == 1) &
                  (pred_labels == 1))[0].shape[0]
    tp_plus_fn = (gt_labels == 1).sum()

    recall = tp / tp_plus_fn * 100.

    print('accuracy = {:.4f}\nrecall = {:.4f}'.format(
        acc, recall))


def results2vinbigdata_cls(results, dst_path, dataset):
    """results (list[tuple[np.array]]): (scores, clses) [bs, ]"""
    img_infos = dataset.img_infos
    img_names = [os.path.splitext(info['filename'])[0] for info in img_infos]
    dst_dir = os.path.dirname(dst_path)
    if not osp.exists(dst_dir):
        os.mkdir(dst_dir)
    dst_path = os.path.splitext(dst_path)[0] + '.csv'

    pred_scores = np.concatenate([res[0] for res in results], axis=0)
    pred_labels = np.concatenate([res[1] for res in results], axis=0)
    with open(dst_path, 'w') as f_out:
        csv_writer = csv.writer(f_out)
        csv_writer.writerow(['image_id', 'class', 'score'])
        for img_id in range(pred_labels.shape[0]):
            score = pred_scores[img_id]
            cls = pred_labels[img_id]
            line = [img_names[img_id], cls, '{:.4f}'.format(score)]
            csv_writer.writerow(line)
    return True


def offline_eval_vinbigdata(save_info_file, score_thr, pre_scale=0.5):
    save_info_dict = mmcv.load(save_info_file)
    results = save_info_dict['outputs']
    gt_annos = save_info_dict['gt_annos']
    eval_vinbigdata_voc(results, gt_annos, pre_scale, score_thr)


if __name__ == '__main__':
    # res_det_csv = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference/yolov4_FirstRun_0126/results_test_cls.csv'
    # sub_sample_csv = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/vinbigdata_chestXray/sample_submission.csv'
    # out_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference/yolov4_FirstRun_0126/results_test_clsFilter.csv'
    # filter_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/data/vinbigdata/annotations/two_cls_test_pred.csv'
    # two_class_filter(res_det_csv, filter_file, sub_sample_csv, out_file)

    root_path = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/mmdetection/work_dirs'
    result_file = os.path.join(root_path, 'DHFRCNN_rx101_64x4d_pafpn_BNmstrain_F1_0119/results/results_val_cls.pkl')
    score_thr = 0.05
    pre_scale = 1.0
    offline_eval_vinbigdata(result_file, score_thr, pre_scale)