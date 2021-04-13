from itertools import zip_longest
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import os
import csv

import sys
sys.path.insert(0, './')
from utils.vinbigdata import read_csv_files, bbox2csvline


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return np.array([list(i) for i in zip_longest(*args, fillvalue=fillvalue)])


def apply_classifier_to_csvRes(pred_csv_file, classifier_info_file, out_file, conf_thr=0.08, safe_PP=False):
    """
    The post-processing method written by 'ShinSiang'
    """
    # read the prediction of detector
    df_sub_b4_cls = pd.read_csv(pred_csv_file)
    # convert it to dataframe format
    d = df_sub_b4_cls.set_index('image_id')['PredictionString'].str.split().apply(grouper, n=6).to_dict()
    df_main = pd.DataFrame(columns=['image_id', 'yolo_class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max'])
    for key in tqdm(d.keys()):
        df_tmp = pd.DataFrame(d[key], columns=['yolo_class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max'])
        df_tmp['image_id'] = key
        df_main = df_main.append(df_tmp, ignore_index=True)
    for i in ['confidence', 'x_min', 'y_min', 'x_max', 'y_max']:
        df_main[i] = df_main[i].astype(float)
    # read the prediction of classifier
    classifier_scores = pd.read_csv(classifier_info_file).rename(
        columns={'target': 'cls_score_mean'}
        # columns={'cls_score': 'cls_score_mean'}
    )
    yolo_dat = df_main.merge(
        classifier_scores,
        on='image_id',
        how='outer'
    )
    print(yolo_dat[:10])
    # for those images that are very likely to be normal image, we lower the conf of the boxes
    # by multiplying it with the classifier score and a low_float_const
    # the low_float_const is to make sure all these boxes have lower confidence than the boxes in other images
    # that is NOT below the classifer threshold
    if safe_PP:
        low_float_const = 0.01
        # for a safe PP
        assert (yolo_dat.loc[yolo_dat['cls_score_mean'] < conf_thr, 'confidence'] * \
                yolo_dat.loc[yolo_dat['cls_score_mean'] < conf_thr, 'cls_score_mean'] * low_float_const).max() < \
               yolo_dat.loc[yolo_dat['cls_score_mean'] < conf_thr, 'confidence'].min()
    else:
        low_float_const = 1
    yolo_dat.loc[yolo_dat['cls_score_mean'] < conf_thr, 'confidence'] = \
        yolo_dat.loc[yolo_dat['cls_score_mean'] < conf_thr, 'confidence'] * \
        yolo_dat.loc[yolo_dat['cls_score_mean'] < conf_thr, 'cls_score_mean'] * low_float_const
    yolo_dat['PredictionString'] = \
        yolo_dat['yolo_class'].astype(str) + " " + yolo_dat['confidence'].astype(str) + " " + \
        yolo_dat['x_min'].astype(str) + " " + yolo_dat['y_min'].astype(str) + " " + \
        yolo_dat['x_max'].astype(str) + " " + yolo_dat['y_max'].astype(str)
    yolo_predstr = yolo_dat.groupby('image_id', as_index=False)['PredictionString'].apply(lambda x: " ".join(x))
    final_output = yolo_predstr.merge(classifier_scores,
                                      on='image_id',
                                      how='outer')
    # Append
    final_output['PredictionString'] = final_output['PredictionString'] + ' 14 ' + (
                1 - final_output['cls_score_mean']).astype(str) + ' 0 0 1 1'
    final_output = final_output[['image_id', 'PredictionString']]
    final_output.to_csv(out_file, index=False)

    print('all done!')


def convert_classifier_info(classifier_info_file, out_file):
    df = pd.read_csv(classifier_info_file)
    df_save = pd.DataFrame()
    df_save['image_id'] = df['image_id'].tolist()
    df_save['target'] = df['classifier_score'].tolist()

    df_save.to_csv(out_file, index=False)
    print('all done!')


def My_apply_classifier_to_csvRes(pred_csv_file, classifier_info_file, out_file, conf_thr=0.08, safe_PP=False):
    """
    The post-processing method written by 'Zehui Gong'
    """
    print('1. reading predictions....')
    pred_img_bboxes, _ = read_csv_files(pred_csv_file)
    df_ClsInfo = pd.read_csv(classifier_info_file)
    imId2ClsScore = {row['image_id']: row['target'] for i, row in df_ClsInfo.iterrows()}

    print('2. start post-processing...')
    if safe_PP:
        low_float_const = 0.01
    else:
        low_float_const = 1.0
    img_ids = []
    pred_strs = []
    for im_id, pred_boxes in tqdm(pred_img_bboxes.items(), total=len(pred_img_bboxes)):
        classifier_score = imId2ClsScore[im_id]
        if classifier_score < conf_thr:
            pred_boxes[:, 4] *= classifier_score * low_float_const
        row = bbox2csvline(pred_boxes, im_id, pre_scale=1.0, score_thr=0)
        pred_str = row[1] + ' 14 {} 0 0 1 1'.format(1 - classifier_score) if row[1] != '14 1 0 0 1 1' else '14 {} 0 0 1 1'.format(1 - classifier_score)
        pred_strs.append(pred_str)
        img_ids.append(row[0])

    print('3. saving processed boxes to out_file...')
    df_save = pd.DataFrame()
    df_save['image_id'] = img_ids
    df_save['PredictionString'] = pred_strs
    df_save.to_csv(out_file, index=False)
    print('all done!')


def apply_2_class_filterV4(pred_csv, out_csv, filter_info_file, thr=0.08):
    """
    The V4 version of applying 2-class filter, only replace the PredString with
    '14 1 0 0 1 1' when classification socre is lower than threshold.
    """
    df_pred = pd.read_csv(pred_csv)
    df_filter = pd.read_csv(filter_info_file)
    pred_strs = df_pred['PredictionString'].tolist()
    img_ids = df_pred['image_id'].tolist()

    num_normal = 0
    for idx in tqdm(range(len(pred_strs))):
        im_id = img_ids[idx]
        cls_score = df_filter[df_filter['image_id'] == im_id]['target'].tolist()[0]
        if cls_score < thr:  # No finding
            pred_strs[idx] = '14 1 0 0 1 1'
            num_normal += 1
    print('number of No finding images: ', num_normal)

    df_save = pd.DataFrame()
    df_save['image_id'] = img_ids
    df_save['PredictionString'] = pred_strs
    df_save.to_csv(out_csv, index=False)
    print('all done!')


def remove_pred_by_specific_file(pred_csv_file, remove_info_file, out_file):
    """written specificly for senyang"""
    df_remove = pd.read_csv(remove_info_file)
    remove_imgIds = set(df_remove['file'].tolist())

    pred_img_bboxes, _ = read_csv_files(pred_csv_file)
    for im_id in list(pred_img_bboxes.keys()):
        if im_id not in remove_imgIds:
            continue
        pred_boxes = pred_img_bboxes[im_id]
        idxes = np.where(pred_boxes[:, -1] == 14)[0]
        pred_boxes = pred_boxes[idxes]
        assert pred_boxes.shape[0] == 1  # only '14 prob 0 0 1 1' left
        if pred_boxes[0, 4] < 0.8:
            pred_boxes[0, 4] = 0.9 + np.random.uniform(0, 1) * 0.09
        pred_img_bboxes[im_id] = pred_boxes

    print('3. saving new predictions into out_file...')
    with open(out_file, 'w') as f_out:
        csv_weiter = csv.writer(f_out)
        csv_weiter.writerow(['image_id', 'PredictionString'])
        for im_name, im_boxes in pred_img_bboxes.items():
            csv_line = bbox2csvline(im_boxes, im_name, pre_scale=1.0, score_thr=0)
            csv_weiter.writerow(csv_line)
    print('all done!')


def get_cls_score(pred_csv_file, remove_info_file, out_file):
    df_remove = pd.read_csv(remove_info_file)
    remove_imgIds = df_remove['file'].tolist()
    pred_img_bboxes, _ = read_csv_files(pred_csv_file)
    scores = []
    for im_id in remove_imgIds:
        if im_id not in pred_img_bboxes:
            raise ValueError
        pred_boxes = pred_img_bboxes[im_id]
        idxes = np.where(pred_boxes[:, -1] == 14)[0]
        pred_boxes = pred_boxes[idxes]
        assert pred_boxes.shape[0] == 1  # only '14 prob 0 0 1 1' left
        scores.append(pred_boxes[0, 4])
    df_remove['normal_score'] = scores
    df_remove.to_csv(out_file, index=False)
    print('all done!')


if __name__ == '__main__':
    # pred_csv_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/' \
    #                 'inference/exp_yolov4_p6_Wnormal_allF3_0208/results_val_cls.csv'
    # classifier_info_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/' \
    #                 'inference/exp_yolov4_p6_Wnormal_allF3_0208/efnb6_oof_ern_sub.csv'
    # out_file = os.path.dirname(pred_csv_file) + '/results_val_PP.csv'
    # conf_thr = 0.04
    # # apply_classifier_to_csvRes(pred_csv_file, classifier_info_file, out_file,
    # #                            conf_thr=conf_thr,
    # #                            safe_PP=False)
    #
    # My_apply_classifier_to_csvRes(
    #     pred_csv_file, classifier_info_file, out_file,
    #     conf_thr=conf_thr,
    #     safe_PP=True)

    # convert_classifier_info(classifier_info_file, os.path.dirname(pred_csv_file) + '/efnb6_oof_ern_sub.csv')


    filter_info_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/' \
                       'VinBigdata_AbnormDetect/processed_data/2_cls_test_pred.csv'
    pred_csv_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference/exp_yolov4_p6_rad_id8F3_0130/results_test_clsbest60_90Final.csv'
    # filter_info_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference/exp_yolov4_p6_Wnormal_allF3_0208/efnb6_oof_ern_sub.csv'
    out_file = os.path.join(os.path.dirname(pred_csv_file), 'results_test_cls_R8F3_PP5.csv')
    # apply_2_class_filterV4(pred_csv=pred_csv_file,
    #                        out_csv=out_file,
    #                        filter_info_file=filter_info_file,
    #                        thr=0.08)
    #
    My_apply_classifier_to_csvRes(
        pred_csv_file, filter_info_file, out_file,
        conf_thr=0.08,
        safe_PP=False)

    # apply_2_class_filterV4(pred_csv_file, out_file, filter_info_file, thr=0.08)

    # pred_csv_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference/exp_yolov4_p7_SenTiff_MyOldallF3_0327/merge_test_tiff_R8F12345_R9F4_R10F1_OldAllF13_P7F13.csv'
    # remove_info_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/test_center/test_0.140.csv'
    # out_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/0.280_nmsnolimit_aortic_cls10_multimul_cls14mul4cls35_multithres13_1024_1280_remove140.csv'
    # remove_pred_by_specific_file(pred_csv_file, remove_info_file, out_file)


    # remove_info_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/test_center'
    # out_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/test_center_280score'
    # os.makedirs(out_dir, exist_ok=True)
    # for file in os.listdir(remove_info_dir):
    #     out_file = os.path.join(out_dir, file)
    #     remove_info_file = os.path.join(remove_info_dir, file)
    #     get_cls_score(pred_csv_file, remove_info_file, out_file)