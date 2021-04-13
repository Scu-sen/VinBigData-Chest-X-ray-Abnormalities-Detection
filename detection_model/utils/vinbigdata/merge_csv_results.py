import os
import csv
import numpy as np
import mmcv
import glob
from ensemble_boxes import weighted_boxes_fusion
import torch
import argparse
import sys
sys.path.insert(0, './')

from utils.vinbigdata.utils import bbox2csvline, read_csv_files, pad_lesion_image_to_sample
from utils.vinbigdata.eval_vinbigdata import eval_from_csv_yolomAP


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def wbf(dets, iou_thr=0.55, conf_type='avg', is_numpy=True):
    """dets (Tensor or np.ndarray): [n, 5] (x1, y1, x2, y2, score)"""
    is_tensor = False
    if isinstance(dets, torch.Tensor):
        device = dets.device
        dets = dets.data.cpu().numpy()
        is_tensor = True
    x1, x2 = dets[:, 0], dets[:, 2]
    y1, y2 = dets[:, 1], dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    new_boxes = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr >= iou_thr)[0]
        overlap_boxes = np.vstack([dets[i:i + 1, :],
                                   dets[order[inds + 1], :]])
        overlap_boxes[:, :4] = overlap_boxes[:, :4] * overlap_boxes[:, 4:]
        box = np.sum(overlap_boxes[:, :4], axis=0) / np.sum(overlap_boxes[:, 4])
        new_boxes.append(box.tolist() + [dets[i, -1]])

        # update order
        inds = np.where(ovr < iou_thr)[0]
        order = order[inds + 1]
    new_boxes = np.array(new_boxes, dtype=np.float32)
    if is_tensor:
        new_boxes = torch.from_numpy(new_boxes).to(device)
    return new_boxes, keep


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.

    Example:
        >>> dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
        >>>                  [49.3, 32.9, 51.0, 35.3, 0.9],
        >>>                  [49.2, 31.8, 51.0, 35.4, 0.5],
        >>>                  [35.1, 11.5, 39.1, 15.7, 0.5],
        >>>                  [35.6, 11.8, 39.3, 14.2, 0.5],
        >>>                  [35.3, 11.5, 39.9, 14.5, 0.4],
        >>>                  [35.2, 11.7, 39.7, 15.7, 0.3]], dtype=np.float32)
        >>> iou_thr = 0.7
        >>> supressed, inds = nms(dets, iou_thr)
        >>> assert len(inds) == len(supressed) == 3
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets.cpu().numpy()
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        dets_th = dets
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        inds = py_cpu_nms(dets_th, iou_thr)
        # dets, inds = wbf(dets_th, iou_thr)
    if not is_numpy:
        inds = torch.from_numpy(np.array(inds, dtype=np.int)).long()
        # dets = torch.from_numpy(dets)
    return dets[inds, :], inds
    # return dets, inds


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = nms
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels


def nms_results(img_bboxes, nms_cfg, img_info_file=None):
    """
    execute Non-maximum suppresseion.
    Args:
        img_bboxes (dict): (n, 6) [x1, y1, x2, y2, score, cls]
        nms_cfg (dict):
    Returns:
        bboxes after NMS: [x1, y1, x2, y2, score, cls]
    """
    # nms_type = nms_cfg['cfg']['type']
    # if nms_type != 'wbf':
    num_classes = nms_cfg['num_classes']
    for im_name in list(img_bboxes.keys()):
        bboxes = img_bboxes[im_name]
        multi_bboxes = bboxes[:, :4].astype(np.float32)
        multi_scores = np.zeros((bboxes.shape[0], num_classes + 1), dtype=np.float32)
        clses = bboxes[:, -1].astype(np.int)
        multi_scores[range(bboxes.shape[0]), clses + 1] = bboxes[:, -2]

        bboxes, labels = multiclass_nms(
            torch.from_numpy(multi_bboxes),
            torch.from_numpy(multi_scores),
            score_thr=nms_cfg['score_thr'],
            nms_cfg=nms_cfg['cfg'],
            max_num=nms_cfg['max_num'])
        img_bboxes[im_name] = np.hstack([bboxes.data.numpy(), labels.data.numpy()[:, np.newaxis]])
    return img_bboxes


def wbf_results(img_bboxes, nms_cfg, img_info_file):
    """
    execute weighted boxes fusion (WBF) to merge the overlapping boxes.
    Args:
        img_bboxes (dict): (n, 6) [x1, y1, x2, y2, score, cls]
        nms_cfg (dict): {score_thr, max_num, num_classes, cfg=dict(type=nms_type, iou_thr=0.5)}
        img_info_file (str): path to load the image height and width
    Returns:
        bboxes after NMS: [x1, y1, x2, y2, score, cls]
    """
    img_infos = {}
    test_info = mmcv.load(img_info_file)
    for im_info in test_info['images']:
        filename = im_info['file_name']
        height = im_info['height']
        width = im_info['width']
        img_infos[os.path.splitext(filename)[0]] = (width, height)

    for im_name in list(img_bboxes.keys()):  # loop through images
        bboxes_list = img_bboxes[im_name]
        boxes_list = []
        scores_list =  []
        labels_list = []
        im_w, im_h = img_infos[im_name]
        for boxes in bboxes_list:
            boxes[:, 0:4:2] /= im_w
            boxes[:, 1:4:2] /= im_h
            boxes_list.append(boxes[:, :4])
            scores_list.append(boxes[:, 4])
            labels_list.append(boxes[:, -1].astype(np.int))
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=nms_cfg['cfg']['iou_thr'],
            skip_box_thr=nms_cfg['score_thr'])
        boxes[:, 0:4:2] *= im_w
        boxes[:, 1:4:2] *= im_h
        img_bboxes[im_name] = np.hstack([boxes, scores[:, np.newaxis], labels[:, np.newaxis]])
    return img_bboxes


def merge_results_from_csv(csv_files, sub_sample_csv, out_file, pre_scale=1.0, score_thr=0.05,
                           nms_cfg=None, split='test', img_info_file=None, anno_file=None):
    """
    convert one(many) pkl results to submit csv format.
    1. read csv files, merge all csv results into one.
    2. NMS, filter overlapped boxes
    3. convert results to csv format
    4. unpad results to original 3000 test image.
    Args:
        csv_files (list or str): path to load csv results, if it is a list, it needs NMS.
        pkl_results is a dict.
        {
            'outputs' (list[list[np.array]]): detection results
            'img_names' (list[str]): image_id
        }
        sub_sample_csv:
        out_file:
        nms_cfg (dict):
        img_info_file (str): file to load image_shape for weightd boxes fusion method
    Returns:
        None
    """
    print('1. read pkl files, merge all pkl results into one.')
    img_bboxes, need_nms = read_csv_files(csv_files, nms_type=nms_cfg['cfg']['type'])

    if need_nms:
        nms_type = nms_cfg['cfg']['type']
        print('2. Using {} to filter overlapped boxes'.format(nms_type.upper()))
        if nms_type != "wbf":
            nms_results(img_bboxes, nms_cfg=nms_cfg)
        else:
            assert img_info_file is not None, "must provide img_info_file when using wbf!"
            wbf_results(img_bboxes, nms_cfg, img_info_file)

    print('3. convert results to csv format')
    with open(out_file, 'w') as f_out:
        csv_weiter = csv.writer(f_out)
        csv_weiter.writerow(['image_id', 'PredictionString'])
        for im_name, im_boxes in img_bboxes.items():
            csv_line = bbox2csvline(im_boxes, im_name, pre_scale, score_thr)
            csv_weiter.writerow(csv_line)
    if split == 'test':
        print('4. unpad results to original 3000 test image.')
        pad_lesion_image_to_sample(out_file, sub_sample_csv, out_file)
    else:
        print('3. this is {} split, we will eval the merged results.'.format(split))
        eval_from_csv_yolomAP(out_file, anno_file)

    print('all done')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='./inference/PreTrained_results', help='path to source csv files')
    parser.add_argument('--out_dir', type=str, default='../vinbigdata_classifierPP_2021/detector_preds', help='path to output directory')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sub_sample_csv = './data/vinbigdata/annotations/sample_submission.csv'
    split = 'test'
    nms_type = 'nms'

    args = parse_args()
    csv_files = glob.glob(os.path.join(args.src_dir, '*.csv'))
    assert len(csv_files) == 11, 'must be 11 csv files'

    out_file = os.path.join(args.out_dir, 'final_before_pp.csv')

    pre_scale = 1.0
    score_thr = 1e-6

    nms_cfg = dict(
        score_thr=score_thr,
        max_num=10000,
        num_classes=15,
        cfg=dict(type=nms_type,
                 iou_thr=0.5 if nms_type != 'wbf' else 0.6))
    print('using {} method to merge results from different models'.format(nms_type))
    merge_results_from_csv(
        csv_files,
        sub_sample_csv,
        out_file,
        pre_scale=pre_scale,
        score_thr=score_thr,
        nms_cfg=nms_cfg,
        split=split)

