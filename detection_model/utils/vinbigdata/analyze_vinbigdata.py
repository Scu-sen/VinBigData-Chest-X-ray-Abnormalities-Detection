import os
import matplotlib.pyplot as plt
import numpy as np
import mmcv
from pycocotools.coco import COCO
import cv2
from itertools import combinations
import glob
from tqdm import tqdm
import seaborn as sns
from multiprocessing import Pool, cpu_count
import pandas as pd

import sys
sys.path.insert(0, './')
from utils.vinbigdata.vinbig_label import Read_csv_annotation
from utils.vinbigdata import load_annotations
from utils.vinbigdata.get_center_data import concact_all_center_csvs


def Imgs_RadIa_Annoted(csv_anno_file, save_dir, num_rads=17, post_fix=''):
    """analyze how many images per radiologist has annotated."""
    if isinstance(csv_anno_file, str):
        annotations, _ = Read_csv_annotation(csv_anno_file, include_radId=True)
    elif isinstance(csv_anno_file, dict):
        annotations = csv_anno_file
    else:
        raise TypeError

    counts = np.zeros((num_rads, ), dtype=np.int)

    num_annotated_imgs = 0
    for im_id, anno in annotations.items():
        anno = np.array(anno, dtype=np.int)
        keep_idxs = np.where(anno[:, 4] != 14)[0]
        if keep_idxs.shape[0] < 1:
            continue
        num_annotated_imgs += 1
        anno = anno[keep_idxs]
        unique_radids = np.unique(anno[:, -1])
        counts[unique_radids-1] += 1

    x = range(num_rads)
    rect = plt.bar(x, height=counts, width=0.5, alpha=0.8, color='blue', label='count')
    plt.xticks(x, labels=['R{}'.format(i+1) for i in range(num_rads)])
    plt.ylabel('number')
    plt.xlabel('rad_ids')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'rad_annos_analyze{}.png'.format(post_fix)))
    plt.close()
    print('num_annoted imgs: ', num_annotated_imgs)
    print('counts: ', counts)


def Ratios_PerCls(csv_anno_file, save_dir, num_class=14, post_fix=''):
    """analyze the ratio of each class appear in the total of images"""
    if isinstance(csv_anno_file, str):
        annotations, _ = Read_csv_annotation(csv_anno_file, include_radId=True)
    elif isinstance(csv_anno_file, dict):
        annotations = csv_anno_file
    else:
        raise TypeError
    ratios = np.zeros((num_class, ), dtype=np.float32)
    num_annotated_imgs = 0
    for im_id, anno in annotations.items():
        anno = np.array(anno, dtype=np.int)
        keep_idxs = np.where(anno[:, 4] != 14)[0]
        if keep_idxs.shape[0] < 1:
            continue
        num_annotated_imgs += 1
        anno = anno[keep_idxs]
        unique_cls = np.unique(anno[:, 4])
        ratios[unique_cls] += 1

    x = range(num_class)
    rect = plt.bar(x, height=(ratios / num_annotated_imgs * 100).astype(np.int),
                   width=0.5, alpha=0.8, color='blue', label='ratios')
    plt.xticks(x, labels=['C{}'.format(i + 1) for i in range(num_class)])
    plt.ylabel('ratios')
    plt.xlabel('class')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'class_ratio_analyze{}.png'.format(post_fix)))
    plt.close()
    print('num_annoted imgs: ', num_annotated_imgs)
    print('ratios: ', ratios)


def center_statistics(csv_anno_file, center_csv_dir, save_dir):
    """Analyze the statistics of each center.
    1. How many image per radologist annotated;
    2. Class distribution in each center.
    """
    mmcv.mkdir_or_exist(save_dir)
    df_center, center_post_fixs = concact_all_center_csvs(center_csv_dir)

    annotations, _ = Read_csv_annotation(csv_anno_file, include_radId=True)
    print('start analyze centers.')
    for ct_idx in tqdm(range(len(center_post_fixs))):
        img_ids = df_center[df_center['center_id'] == ct_idx]['image_id'].tolist()
        sub_annos = {im_id: annotations[im_id] for im_id in img_ids}
        Imgs_RadIa_Annoted(sub_annos.copy(), save_dir, post_fix=center_post_fixs[ct_idx])
        Ratios_PerCls(sub_annos.copy(), save_dir, post_fix=center_post_fixs[ct_idx])
    print('all done!')


def NumAnnos_PerImg(csv_anno_file):
    """class weights"""
    annotations, _ = Read_csv_annotation(csv_anno_file)
    num_annotated_imgs = 0
    labels = []
    for im_id, anno in annotations.items():
        anno = np.array(anno, dtype=np.int)
        keep_idxs = np.where(anno[:, 4] != 14)[0]
        if keep_idxs.shape[0] < 1:
            continue
        num_annotated_imgs += 1
        anno = anno[keep_idxs]
        # total_annos[0] += anno.shape[0]
        labels.append(anno[:, 4])
    classes = np.hstack(labels).astype(np.int)
    weights = np.bincount(classes, minlength=14)
    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    print(weights)


def analyze_lesion_location(anno_file, save_dir, binary_thr, debug=False):
    """analyze the possible location of each lesion category."""
    mmcv.mkdir_or_exist(save_dir)
    print('1. get bboxes')
    coco = COCO(anno_file)
    img_ids = coco.getImgIds()
    img_infos = [coco.loadImgs([i])[0] for i in img_ids]
    image_bboxes = []
    for i in range(len(img_infos)):
        img_id = img_infos[i]['id']
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        ann_info = coco.loadAnns(ann_ids)
        bboxes = []
        for i, ann in enumerate(ann_info):
            if ann['category_id'] == 15:
                continue
            x1, y1, w, h = ann['bbox']
            x2, y2 = x1 + w, y1 + h
            cls = ann['category_id']
            bboxes.append([x1, y1, x2, y2, cls])
        if bboxes:
            bboxes = np.array(bboxes, dtype=np.float32)
        else:
            bboxes = np.zeros((0, 5), dtype=np.float32)
        bboxes[:, 0:4:2] /= img_infos[i]['width']
        bboxes[:, 1:4:2] /= img_infos[i]['height']
        image_bboxes.append(bboxes)

    print('2. generate heatmap for each of the classes')
    heat_width, heat_height = 400, 500
    num_classes = 14
    # 1-based cls
    image_bboxes = np.vstack(image_bboxes)
    image_bboxes[:, 0:4:2] *= heat_width
    image_bboxes[:, 1:4:2] *= heat_height
    heatmaps = np.zeros((num_classes, heat_height, heat_width), dtype=np.float32)
    for box_i in range(image_bboxes.shape[0]):
        x1, y1, x2, y2, cls = image_bboxes[box_i].astype(np.int)
        heatmaps[cls-1, y1:y2, x1:x2] += 1
    max_vals = np.max(heatmaps, axis=(1, 2), keepdims=True)
    min_vals = np.min(heatmaps, axis=(1, 2), keepdims=True)
    heatmaps = (heatmaps - min_vals) / (max_vals - min_vals)
    heatmaps *= 255

    if debug:
        print('3. visualize the heatmaps')
        classes = [
            "0 - Aortic enlargement",
            "1 - Atelectasis",
            "2 - Calcification",
            "3 - Cardiomegaly",
            "4 - Consolidation",
            "5 - ILD",
            "6 - Infiltration",
            "7 - Lung Opacity",
            "8 - Nodule/Mass",
            "9 - Other lesion",
            "10 - Pleural effusion",
            "11 - Pleural thickening",
            "12 - Pneumothorax",
            "13 - Pulmonary fibrosis",
            "14 - No finding",
        ]
        fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(20, 30))
        for i, ax in enumerate(axs.flatten()):
            if i >= num_classes:
                break
            ax.imshow(heatmaps[i].astype(np.uint8),
                      cmap='hot',
                      interpolation='nearest')
            _ = ax.set_title(classes[i], fontweight="bold", size=15)
        plt.savefig(os.path.join(save_dir, 'lesion_location_vis.png'))
        plt.close()

    print('4. get clsaa-wise lesion location boxes')
    class_lesion_boxes = dict()
    binary_heats = []
    for cls in range(num_classes):
        thr = binary_thr if isinstance(binary_thr, int) else binary_thr[cls]
        binary_cls_heat = (heatmaps[cls] > thr).astype(np.uint8)
        binary_heats.append(binary_cls_heat)
        # boxes_info: [x, y, w, h, area, ctx, cty]
        num_comp, comp, stat, centroid = cv2.connectedComponentsWithStats(binary_cls_heat, connectivity=8)
        boxes_infos = np.hstack([stat, centroid])
        ovr = boxes_infos[:, -3] / (heat_width * heat_height)
        keep_idxes = np.where((ovr < 0.8) & (ovr >= 0.01))[0]
        class_lesion_boxes[cls] = boxes_infos[keep_idxes]
    if debug:
        print('visualize lesion location bounding boxes.')
        fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(20, 30))
        for i, ax in enumerate(axs.flatten()):
            if i >= num_classes:
                break
            bboxes = class_lesion_boxes[i][:, :4]
            bboxes[:, 2:4] = bboxes[:, 0:2] + bboxes[:, 2:4]
            heat_vis = binary_heats[i] * 255
            img_vis = np.repeat(heat_vis[:, :, np.newaxis], 3, axis=2)
            img_vis = mmcv.imshow_bboxes(img_vis, bboxes, colors='green', top_k=-1, thickness=1, show=False)
            ax.imshow(img_vis, interpolation='nearest')
            _ = ax.set_title(classes[i], fontweight="bold", size=15)
        plt.savefig(os.path.join(save_dir, 'lesion_location_binary_vis.png'))

    print('5. Normalizing class lesion boxes.')
    for key in list(class_lesion_boxes.keys()):
        boxes = class_lesion_boxes[key]
        # (x1, y1, x2, y2, area, ctx, cty)  Normalized
        boxes[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]
        boxes[:, 0:4:2] /= heat_width
        boxes[:, 1:4:2] /= heat_height
        boxes[:, -2] /= heat_width
        boxes[:, -1] /= heat_height
        class_lesion_boxes[key] = boxes
    print('saving class_lesion_boxes into file')
    mmcv.dump(class_lesion_boxes, os.path.join(save_dir, 'class_lesion_boxes.pkl'))
    print('all done!')


def analyze_cooccurrence_of_classes(json_anno_file, out_file, num_classes=14):
    """
    analyze the co-occurrence of 14 classes, the result is a two-dimension matrix.
    """
    classes = [
        "Aortic enlargement",
        "Atelectasis",
        "Calcification",
        "Cardiomegaly",
        "Consolidation",
        "ILD",
        "Infiltration",
        "Lung Opacity",
        "Nodule/Mass",
        "Other lesion",
        "Pleural effusion",
        "Pleural thickening",
        "Pneumothorax",
        "Pulmonary fibrosis"]

    mmcv.mkdir_or_exist(os.path.dirname(out_file))
    annotations = load_annotations(json_anno_file)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int)
    for im_id, annos in annotations.items():  # compute the co-occurrence matrix
        clses = np.unique(annos[:, 0].astype(np.int)).tolist()
        for comb in combinations(clses, 2):
            matrix[comb[0], comb[1]] += 1
            matrix[comb[1], comb[0]] += 1

    # plot the heatmap
    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min() * 1.0)
    Thickening_occur = np.argsort(-matrix[11, :])
    print('co-occurrence for Pleural thickening: ', [classes[i] for i in Thickening_occur])
    # plt.figure(dpi=300)
    # sns.heatmap(data=matrix,
    #             cmap=sns.cubehelix_palette(as_cmap=True),
    #             xticklabels=[classes[i] for i in range(num_classes)],
    #             yticklabels=[classes[i] for i in range(num_classes)])
    # plt.title('The co-occurrence of the classes.')
    # plt.tight_layout()
    # plt.savefig(out_file)
    # plt.close()
    # print('all done!')


def check_normal_abnormal_num(info_file, json_anno_file):
    df_info = pd.read_csv(info_file)
    img_names = df_info['file'].tolist()
    is_abnormal = []

    annotations = load_annotations(json_anno_file)
    img_names_withAnno = set(os.path.splitext(im_name)[0] for im_name in annotations.keys())

    normal_num = 0
    abnormal_num = 0
    for name in img_names:
        if name in img_names_withAnno:
            abnormal_num += 1
            is_abnormal.append(1)
        else:
            normal_num += 1
            is_abnormal.append(0)
    df_info['is_abnormal'] = is_abnormal
    df_info.to_csv(info_file)
    print('normal_num: {}  abnormal_num:{}'.format(normal_num, abnormal_num))


def get_max_min(file, thr):
    img = cv2.imread(file)
    return (img.max() - img.min()) > thr


def count_abnormal_ImgNums(data_path, std_thr):
    """
    counting the number of abnormal images in the vinbigdata dataset
    """
    num_abnormal_imgs = 0
    all_imgs = glob.glob(os.path.join(data_path, '*.png'))

    pools = Pool(cpu_count())
    async_results = []
    for img_file in tqdm(all_imgs):
        async_results.append(
            pools.apply_async(get_max_min, (img_file, std_thr))
        )
    for async_res in async_results:
        async_res.wait()
        res = async_res.get()
        if res:
            num_abnormal_imgs += 1

    print('there are total {} abnormal images in dataset'.format(num_abnormal_imgs))
    return


def get_abnormal_ornot():
    all_info_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/multilabel_cls_train.csv'
    csv_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/data_center'
    out_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/data_center_out'

    mmcv.mkdir_or_exist(out_dir)
    df_all = pd.read_csv(all_info_file)
    for file in os.listdir(csv_dir):
        df = pd.read_csv(os.path.join(csv_dir, file))
        img_ids = df['file'].tolist()
        sub_df = df_all[df_all['image_id'].isin(img_ids)]
        sub_df.to_csv(os.path.join(out_dir, file), index=False)
    print('all done')


def num_imges_otherRad_Anno(csv_anno_file):
    """
    Counting the number of images annotated by other radiologists, e.g., R11~R17.
    """
    all_annos, _ = Read_csv_annotation(csv_anno_file, include_radId=True)
    num_imgs = 0
    num_abnormal = 0
    rad8910 = set((8, 9, 10))
    for im_id, annos in tqdm(all_annos.items(), total=len(all_annos)):
        bboxes = np.array(annos)
        keep_idxes = np.where(bboxes[:, 4] != 14)[0]
        if keep_idxes.shape[0] < 1:
            continue
        num_abnormal += 1
        bboxes = bboxes[keep_idxes]
        rad_ids = set(bboxes[:, 5].tolist())
        if len(rad8910.intersection(rad_ids)) > 1:  # there is one of rad 8 9 10 in annos
            continue
        num_imgs += 1
    print('num_abnormal: {}'.format(num_abnormal))
    print('there are total {} images annotated by other rads(except rad 8 9 10)'.format(num_imgs))


if __name__ == '__main__':
    csv_anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/vinbigdata_chestXray/train.csv'
    # save_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/center_analysis'
    # center_csv_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/data_center'
    # center_statistics(csv_anno_file, center_csv_dir, save_dir)

    # Ratios_PerCls(csv_anno_file, save_dir)
    # NumAnnos_PerImg(csv_anno_file)

    # anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/vinbig_train.json'
    # save_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/lesion_location'
    # analyze_lesion_location(anno_file, save_dir,
    #                         binary_thr=[130] * 8 + [150] + [130] * 5,
    #                         debug=True)
    # anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/new_Kfold_annotations/rad_id9/vinbig_train_fold4.json'
    # save_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/class_occurrence'
    # analyze_cooccurrence_of_classes(
    #     anno_file, out_file=os.path.join(save_dir, 'occurrence_R9TrainF4.png'))

    # img = mmcv.imread('/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/train/ff924bcbd38f123aec723aa7040d7e43.png')
    # print('max-min: ', img.max() - img.min())
    # data_path = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/test'
    # count_abnormal_ImgNums(data_path, std_thr=30)

    # info_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/train_number.csv'
    # json_anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/vinbig_train.json'
    # check_normal_abnormal_num(info_file, json_anno_file)

    # get_abnormal_ornot()

    num_imges_otherRad_Anno(csv_anno_file)


