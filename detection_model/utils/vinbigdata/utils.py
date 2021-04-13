import pandas as pd
import numpy as np
from functools import partial
import csv


def bbox2csvline(bboxes, img_name, pre_scale, score_thr=0.05):
    """
    Args:
        bboxes (np.array): (n, 6) [x1, y1, x2, y2, score, cls]
    Returns:
    """
    csv_line = [img_name, ]
    idxes = np.argsort(-bboxes[:, -2])
    bboxes = bboxes[idxes]
    idxes = np.where(bboxes[:, -2] >= score_thr)[0]
    bboxes = bboxes[idxes]
    bboxes[:, :4] /= pre_scale

    num_box = bboxes.shape[0]
    # No boxes, Normal slice
    if num_box < 1:
        print('{} have no boxes'.format(img_name))
        res_str = '14 1 0 0 1 1'
    else:
        res_str = ''
        for idx, bbox in enumerate(bboxes):
            score = bbox[-2]
            class_id = int(bbox[-1])
            bbox = np.round(bbox[:4]).astype(np.int)
            temp_txt = '{} {} {} {} {} {}{}'.format(
                class_id, score, bbox[0], bbox[1],
                bbox[2], bbox[3], ' ' if idx < num_box - 1 else '')
            res_str += temp_txt
    csv_line.append(res_str)
    return csv_line


def read_csv(csv_file):
    """
    Read_csv file
    Args:
        csv_file:
    Returns:
    Dict{
        'image_id': det_res
    }
    """
    results = dict()
    df = pd.read_csv(csv_file)
    for idx, row in df.iterrows():
        results[row['image_id']] = row['PredictionString']
    return results


def pad_lesion_image_to_sample(res_det_csv, sub_sample_csv, out_file):
    res_det = read_csv(res_det_csv)
    sample_sub = read_csv(sub_sample_csv)
    sample_sub.update(res_det)

    with open(out_file, 'w') as fout:
        csv_weiter =csv.writer(fout)
        csv_weiter.writerow(['image_id', 'PredictionString'])
        for img_id, res_str in sample_sub.items():  # loop through images
            line = [img_id, res_str]
            csv_weiter.writerow(line)

    print('done!')


def read_csv_files(pkl_files, KeepNoPredImgs=False, nms_type='nms'):
    """return format:
    Args:
        nms_type (str): indicate what type of fusing boxes method we are using,
                        if is "wbf", keep the list type of different files.
    results:
     nms or soft_nms: {'image_id': np.array (k, 6)[x1, y1, x2, y2, score, cls]}
     wbf: {'image_id': list[np.ndarray][x1, y1, x2, y2, score, cls](k, 6)}
    """

    def read_one_file(file, KeepNoPredImgs):
        df = pd.read_csv(file)
        if not KeepNoPredImgs:
            df = df[df['PredictionString'] != '14 1 0 0 1 1']
        img_bboxes_dict = dict()
        for idx, row in df.iterrows():
            img_id = row['image_id']
            box_info = list(map(float, row['PredictionString'].split(' ')))
            clses = np.array(box_info[0::6])
            scores = np.array(box_info[1::6])
            boxes = np.array([box_info[2::6], box_info[3::6],
                              box_info[4::6], box_info[5::6]]).transpose([1, 0])

            boxes = np.hstack([boxes, scores[:, np.newaxis], clses[:, np.newaxis]])
            img_bboxes_dict[img_id] = boxes
        return img_bboxes_dict

    if isinstance(pkl_files, str):
        return read_one_file(pkl_files, KeepNoPredImgs), False

    elif isinstance(pkl_files, (list, tuple)):
        print('merging {} files...'.format(len(pkl_files)))
        all_img_boxes = list(map(partial(read_one_file, KeepNoPredImgs=KeepNoPredImgs), pkl_files))
        img_bboxes = dict()
        for img_name in list(all_img_boxes[0].keys()):
            img_res = []
            for file_i in range(len(pkl_files)):
                img_box = all_img_boxes[file_i].get(img_name, None)
                if img_box is None:
                    print('{}th file img {} have no box'.format(file_i, img_name))
                    img_box = np.zeros((0, 6), dtype=np.float32)
                img_res.append(img_box)
            # wbf need to pass the list of pred_boxes
            if nms_type != "wbf":
                img_bboxes[img_name] = np.vstack(img_res)
            else:
                img_bboxes[img_name] = img_res
        return img_bboxes, True