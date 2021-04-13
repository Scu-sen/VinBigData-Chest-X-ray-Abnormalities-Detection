import torch
import numpy as np
import os
from pycocotools.coco import COCO

import sys
sys.path.pop(0)
sys.path.insert(0, './')
from utils.vinbigdata import read_csv_files, eval_vinbigdata_voc
from utils.general import ap_per_class, box_iou, bbox2result


VinBigData_class_names = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
    'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
    'Pulmonary fibrosis', 'No finding']


def load_annotations(anno_file, keep_no_annos=False):
    """
    Returns:
        {'image_id': np.array (k, 5) [cls_id, x1, y1, x2, y2]}
    """
    coco = COCO(anno_file)
    cat_ids = coco.getCatIds()
    cat2label = {cat_id: i
                 for i, cat_id in enumerate(cat_ids)}
    img_ids = coco.getImgIds()
    labels = {}
    for i in img_ids:  # loop through images
        info = coco.loadImgs([i])[0]
        filename = os.path.splitext(info['file_name'])[0]
        img_id = info['id']
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        ann_info = coco.loadAnns(ann_ids)
        cur_label = []
        for ann in ann_info:  # loop through objects in images
            if ann.get('ignore', False):
                continue
            x1, y1, bw, bh = ann['bbox']
            if ann['area'] <= 0 or bw < 1 or bh < 1:
                continue
            if ann['category_id'] == 15:  # No finding
                continue
            cls_id = cat2label[ann['category_id']]
            box = [cls_id, x1, y1, x1 + bw, y1 + bh]
            cur_label.append(box)
        if cur_label:  # images with annos
            labels[filename] = np.array(cur_label, dtype=np.float32)
        elif keep_no_annos:
            labels[filename] = np.array([[cat2label[15], 0, 0, 1, 1]], dtype=np.float32)
    return labels


def eval_from_csv_yolomAP(pred_csv, anno_file, num_classes=14, KeepNoAnnoImgs=False):
    """
    evaluate the mAP0.5 from pred_csv file
    Args:
        pred_csv (str): file to load the pred result
        anno_file (str): file to load annotation in json coco format
    """
    print('loading prediction results...')
    pred_img_bboxes, _ = read_csv_files(pred_csv, KeepNoPredImgs=KeepNoAnnoImgs)
    print('loading corresponding annotations...')
    gt_img_bboxes = load_annotations(anno_file, keep_no_annos=KeepNoAnnoImgs)
    print('{} images with prediction\n{} images with ground-truth boxes'.format(
        len(pred_img_bboxes), len(gt_img_bboxes)))

    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    stats, ap, ap_class = [], [], []
    seen = 0
    for im_id in list(gt_img_bboxes.keys()):  # statistics per image
        labels = gt_img_bboxes[im_id]
        pred = pred_img_bboxes.get(im_id)

        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        seen += 1
        if pred is None:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue
        pred = torch.from_numpy(pred.astype(np.float32))
        labels = torch.from_numpy(labels)

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = labels[:, 1:5]

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=num_classes)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if num_classes > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (VinBigData_class_names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    print('all done!')


def eval_from_csv_vocmAP(pred_csv, anno_file, num_classes=14, pre_scale=1.0, score_thr=0.001):
    """
    evaluate the mAP0.4 from pred_csv file, using VOC eval mAP method
    Args:
        pred_csv (str): file to load the pred result
        anno_file (str): file to load annotation in json coco format
    """
    pred_img_bboxes, _ = read_csv_files(pred_csv)
    gt_img_bboxes = load_annotations(anno_file)

    results = []
    gt_annos = []
    for im_id in list(gt_img_bboxes.keys()):
        gts = gt_img_bboxes[im_id]
        pred_boxes = pred_img_bboxes.get(im_id)
        if pred_boxes is None:
            pred_boxes = [np.zeros((0, 5), dtype=np.float32)
                          for _ in range(num_classes)]
        else:
            pred_boxes = bbox2result(pred_boxes[:, :-1], pred_boxes[:, -1].astype(np.int),
                                     num_classes=num_classes + 1)
        gts = np.hstack([gts[:, 1:], gts[:, 0:1]])
        results.append(pred_boxes)
        gt_annos.append(gts)

    eval_vinbigdata_voc(results, gt_annos, pre_scale, score_thr)
    print('all done')


if __name__ == '__main__':
    from utils.vinbigdata.vinbig_cvt_res_utils import results2vinbigdata_twoThr
    rad_id = 'rad_id8'
    fold = 1
    pred_csv = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference/exp_yolov4_p6_rad_id8F1_0129/results_val_cls_best60_99Aug.csv'
    # pred_csv = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference/exp_yolov4_p6_SpeCls111214_rad_id9F4_0311/results_val_best_mappedOri.csv'
    anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/' \
                'new_Kfold_annotations_WithNormal/{}/vinbig_val_fold{}.json'.format(rad_id, fold)
    # anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/' \
    #             'processed_data/Kfold_annotations/vinbig_val_fold{}.json'.format(fold)
    # pred_csv = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/inference/exp_yolov4_p6_C13_bothAN_rad_id8F3_0207/merge_results_thr0.001_speCls.csv'
    # anno_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/data/siim_acr/annotations/mergedAnnos_val_rad8F3_cls13.json'

    # for high_thr in np.linspace(0.3, 0.5, 5):
    #     print('high_thr: {}'.format(high_thr))
    #     two_thr_out_file = os.path.join(os.path.dirname(pred_csv), 'results_val_cls_twoThr.csv')
    #     results2vinbigdata_twoThr(pred_csv, two_thr_out_file, high_thr=high_thr, low_thr=0.001)
    #     eval_from_csv_yolomAP(two_thr_out_file, anno_file, num_classes=15, KeepNoAnnoImgs=True)

    eval_from_csv_yolomAP(pred_csv, anno_file, KeepNoAnnoImgs=False)