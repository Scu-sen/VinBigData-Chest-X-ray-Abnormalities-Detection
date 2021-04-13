import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from ensemble_boxes import *
import argparse

def box_ensembling(image_id,
                   annotations_rad1,
                   annotations_rad2,
                   annotations_rad3,
                   iou_thr = 0.5,
                   skip_box_thr = 0.00001):

    # boxes per image_id
    image_boxes1 = annotations_rad1.loc[annotations_rad1['image_id'] == image_id].reset_index(drop = True)
    image_boxes2 = annotations_rad2.loc[annotations_rad2['image_id'] == image_id].reset_index(drop = True)
    image_boxes3 = annotations_rad3.loc[annotations_rad3['image_id'] == image_id].reset_index(drop = True)


    boxes_list = []
    scores_list = []
    labels_list = []
    weights = [1, 1, 1] # all radiologists weighted equally (For now)

    # rad1
    boxes_list.append(np.array(image_boxes1[['x_min_norm', 'y_min_norm', 'x_max_norm', 'y_max_norm']]).tolist())
    labels_list.append(np.array(image_boxes1['class_id']).tolist())

    scores_list.append([0.95] * image_boxes1.shape[0])

    # rad2
    boxes_list.append(np.array(image_boxes2[['x_min_norm', 'y_min_norm', 'x_max_norm', 'y_max_norm']]).tolist())
    labels_list.append(np.array(image_boxes2['class_id']).tolist())

    scores_list.append([0.95] * image_boxes2.shape[0])

    # rad3
    boxes_list.append(np.array(image_boxes3[['x_min_norm', 'y_min_norm', 'x_max_norm', 'y_max_norm']]).tolist())
    labels_list.append(np.array(image_boxes3['class_id']).tolist())

    scores_list.append([0.95] * image_boxes3.shape[0])


    wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(boxes_list,
                                                              scores_list,
                                                              labels_list,
                                                              weights=weights,
                                                              iou_thr=iou_thr,
                                                              skip_box_thr=skip_box_thr)

    # Return data frame
    ret_df = pd.DataFrame.from_records(wbf_boxes, columns = ['x_min_norm', 'y_min_norm',
                                                             'x_max_norm', 'y_max_norm'])
    ret_df['image_id'] = image_id
    ret_df['class_id'] = wbf_labels.astype(int)
    ret_df['scores'] = wbf_scores # just to estimate the merges

    return ret_df


def rad_dict(rad_list):
    dict_ = {}
    id_ = 1

    for rad in rad_list:
        dict_[rad] = id_
        id_ += 1

    return dict_



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data', help='path to data directory')
    parser.add_argument('--iou-thr', type=float, default=0.0001, help='IOU threshold for WBF')
    parser.add_argument('--output-file', type=str, default='data/input/nonR8R9R10_wbf.csv', help='output filename of wbf-merged boxes')
    opt = parser.parse_args()

    train_annotations =  pd.read_csv(os.path.join(opt.data_dir, 'input/train.csv'))
    train_meta = pd.read_csv(os.path.join(opt.data_dir, 'input/train_meta.csv'))
    train_annotations = train_annotations.merge(train_meta, how = 'left', on = 'image_id')

    train_annotations = train_annotations.loc[train_annotations['class_id'] != 14].reset_index(drop = True)

    train_annotations['x_min_norm'] = train_annotations['x_min'] / train_annotations['width']
    train_annotations['x_max_norm'] = train_annotations['x_max'] / train_annotations['width']
    train_annotations['y_min_norm'] = train_annotations['y_min'] / train_annotations['height']
    train_annotations['y_max_norm'] = train_annotations['y_max'] / train_annotations['height']

    # Consolidate non R8 R9 R10 radiologists
    secondary_rad_image_ids = train_annotations.loc[train_annotations['rad_id'].isin(['R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17'])]['image_id'].unique()

    train_annotations = train_annotations.loc[train_annotations['image_id'].isin(secondary_rad_image_ids)].reset_index(drop = True)

    rad_groupings = train_annotations.groupby(['image_id'])['rad_id'].unique().reset_index()
    rad_groupings['rad_id'] = rad_groupings['rad_id'].apply(lambda x: sorted(x, key = lambda x: int(x[1:])))

    rad_groupings['mapping'] = rad_groupings['rad_id'].apply(rad_dict)

    train_annotations = train_annotations.merge(rad_groupings[['image_id', 'mapping']],
                                                how  = 'left',
                                                on = 'image_id')

    train_annotations['rad_id_int'] = train_annotations[['rad_id', 'mapping']].apply(lambda x: x.mapping[x.rad_id], axis = 1)

    train_annotations = train_annotations.loc[train_annotations['class_id'] != 14].reset_index(drop = True)

    train_annotations_R8 = train_annotations.loc[train_annotations['rad_id_int'] == 1].reset_index(drop = True)
    train_annotations_R9 = train_annotations.loc[train_annotations['rad_id_int'] == 2].reset_index(drop = True)
    train_annotations_R10 = train_annotations.loc[train_annotations['rad_id_int'] == 3].reset_index(drop = True)

    imageids = np.intersect1d(np.intersect1d(train_annotations_R8.image_id,
                                             train_annotations_R9.image_id),
                                             train_annotations_R10.image_id)

    # Box ensembling wrapper for multiprocessing
    def box_ensembling_wrap(i):
        return box_ensembling(imageids[i],
                              annotations_rad1 = train_annotations_R8,
                              annotations_rad2 = train_annotations_R9,
                              annotations_rad3 = train_annotations_R10,
                              iou_thr = opt.iou_thr)

    pool = mp.Pool(mp.cpu_count())
    with pool as p:
        res = list(tqdm(p.imap(box_ensembling_wrap,
                               range(len(imageids))),
                               total = len(imageids)))
    pool.terminate()
    pool.join()


    train_annotations_wbf = pd.concat(res).reset_index(drop = True)
    train_annotations_wbf.to_csv(opt.output_file, index = False)
