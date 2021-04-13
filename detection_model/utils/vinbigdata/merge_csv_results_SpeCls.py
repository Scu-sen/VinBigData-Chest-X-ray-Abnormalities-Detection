"""
Author: ZehuiGong (zehuigong@foxmail.com)
Date: 2021-02-06
"""
import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

sys.path.pop(0)
sys.path.insert(0, './')
from utils.vinbigdata.merge_csv_results import read_csv_files, pad_lesion_image_to_sample
from utils.vinbigdata.vinbig_cvt_res_utils import bbox2csvline
from utils.vinbigdata.eval_vinbigdata import eval_from_csv_yolomAP


def merge_results_from_csv_SpeCls(csv_files, sub_sample_csv, out_file, pre_scale=1.0,
                                  pad_cls=12, score_thr=0.05, split='test', mode=('from', 'to')):
    """
    merge one(or many) csv results to submit csv format.
    1. read csv files;
    2. extract boxes of 'spe_cls' from "from csv file" and, pad them to "to csv file";
    3. convert results to submission csv format
    4. unpad results to original 3000 test image (if needed).
    Args:
        csv_files (list or str): path to load csv results.
        sub_sample_csv (str):
        out_file (str): path to save merged results
        pad_cls (int): The class we want to merge.
        mode (list or tuple): flag indicates that which file in csv_files is 'form' file, which is 'to' file.
    Returns:
        None
    """
    print('1. read csv files...')
    all_img_bboxes = [read_csv_files(file)[0] for file in csv_files]
    from_img_boxes = {}
    for idx, flag in enumerate(mode):
        if flag == 'from':  # extract boxes
            from_img_boxes = all_img_bboxes[idx]
        elif flag == 'to':  # and pad boxes
            to_img_boxes = all_img_bboxes[idx]
        else:
            raise ValueError("expected from | to, but got {}".format(flag))

    print('2. extract boxes and paste them into to file...')
    for im_name in tqdm(list(to_img_boxes.keys()), total=len(to_img_boxes.keys())):
        to_boxes = to_img_boxes[im_name]
        from_boxes = from_img_boxes.get(im_name, None)
        if from_boxes is None:
            from_boxes = np.zeros((0, 6), dtype=np.float32)

        # remove original boxes in to_boxes
        keep_idxes = np.where(to_boxes[:, -1] == pad_cls)[0]
        to_boxes = to_boxes[keep_idxes]

        # keep_idxes = np.where(from_boxes[:, -1] == pad_cls)[0]
        # from_boxes = from_boxes[keep_idxes]
        # im_boxes = np.vstack([from_boxes, to_boxes])

        im_boxes = to_boxes
        to_img_boxes[im_name] = im_boxes

    print('3. saving new predictions into out_file...')
    with open(out_file, 'w') as f_out:
        csv_weiter = csv.writer(f_out)
        csv_weiter.writerow(['image_id', 'PredictionString'])
        for im_name, im_boxes in to_img_boxes.items():
            csv_line = bbox2csvline(im_boxes, im_name, pre_scale, score_thr)
            csv_weiter.writerow(csv_line)

    if split == 'test':
        print('4. unpad results to original 3000 test image.')
        pad_lesion_image_to_sample(out_file, sub_sample_csv, out_file)

    print('all done!')


def add_nofinding_to_each_pred(pred_csv, out_csv):
    """Add '14 1 0 0 1 1' to each line of csv prediction,
     to keep the performance of 'No finding' category unchanged (0.052)."""
    df_pred = pd.read_csv(pred_csv)
    pred_strs = df_pred['PredictionString'].tolist()
    # add (14 1 0 0 1 1)
    for idx in range(len(pred_strs)):
        if pred_strs[idx] != '14 1 0 0 1 1':
            pred_strs[idx] += ' 14 1 0 0 1 1'

    df_save = pd.DataFrame()
    df_save['image_id'] = df_pred['image_id'].tolist()
    df_save['PredictionString'] = pred_strs
    df_save.to_csv(out_csv, index=False)
    print('all done!')


def map_cls_backto_original(src_pred_file, map_dict, out_file):
    """
    if we train the specialized detector, we should map the clsid of specialized
    detector(SD) back to original clsid in the dataset.
    Args:
        src_pred_file (str): path to load the output of specialized detector, in csv format
        map_dict (dict): a dict to map the clsid of SD to original clsid in the dataset.
        out_file (str): path to save the mapped result.
    """
    sd_pred_img_boxes = read_csv_files(src_pred_file)[0]
    print('1. mapping clsid back to original')
    for im_id in tqdm(list(sd_pred_img_boxes.keys()), total=len(sd_pred_img_boxes)):
        img_boxes = sd_pred_img_boxes[im_id]
        clses = img_boxes[:, -1].astype(np.int)
        clses = np.array([map_dict[cls] for cls in clses])
        img_boxes[:, -1] = clses

    print('2. saving new predictions into out_file...')
    with open(out_file, 'w') as f_out:
        csv_weiter = csv.writer(f_out)
        csv_weiter.writerow(['image_id', 'PredictionString'])
        for im_name, im_boxes in sd_pred_img_boxes.items():
            csv_line = bbox2csvline(im_boxes, im_name, 1.0, 1e-5)
            csv_weiter.writerow(csv_line)
    print('all done!')
    return


if __name__ == '__main__':
    # sample_submission_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/vinbigdata_chestXray/sample_submission.csv'
    # split = 'test'
    # sub_sample_csv = sample_submission_file
    #
    # csv_files = [
    #     '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference/exp_yolov4_p6_SpeCls111214_rad_id9F4_0311/results_test_clsbest60_119Aug_mapped.csv',
    #     # '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference/exp_yolov4_p6_C13_bothAN_rad_id8F3_0207/results_test_cls.csv'
    # ]
    # out_file = os.path.join(os.path.dirname(csv_files[-1]), 'merge_results_{}_thr0.001_speCls.csv'.format(split))
    #
    #
    # merge_results_from_csv_SpeCls(csv_files,
    #                               sub_sample_csv,
    #                               out_file,
    #                               pre_scale=1.0,
    #                               pad_cls=10,
    #                               score_thr=0.001,
    #                               split=split,
    #                               mode=('to', ))
    #
    # add_nofinding_to_each_pred(out_file,
    #                            # os.path.dirname(csv_files[0]) + '/merge_results_test_thr0.001_speCls.csv'
    #                            out_file)

    src_pred_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference/exp_yolov4_p6_SenTiff_SpeILD_rad_id8F3_0328/results_val_clsbest50_150_199Aug.csv'
    map_dict = {0: 5}
    out_file = os.path.join(os.path.dirname(src_pred_file), 'results_val_clsbest_mapped.csv')
    map_cls_backto_original(src_pred_file, map_dict, out_file)

    rad_id = 'rad_id8'
    fold = 3
    anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/' \
                'new_Kfold_annotations_WithNormal/{}/vinbig_val_fold{}.json'.format(rad_id, fold)
    eval_from_csv_yolomAP(out_file, anno_file)