import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from ensemble_boxes import *



# From https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data



# Return coordinates of the intersection of 2 input bounding boxes
def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return ()
    return (x, y, w, h)


def count_boxes_intersect(boxes, min_count = 3):
    if not boxes:
        return []

    count_results = []

    boxIndex = 0

    while boxIndex < len(boxes):

        count_intersections = 1 # Start with 1 (include the current box as an intersection)

        a = boxes[boxIndex]

        listBoxes = np.delete(boxes, boxIndex, 0)

        for b in listBoxes:
            if intersection(a, b):
                count_intersections = count_intersections + 1

        count_results.append(count_intersections)
        boxIndex = boxIndex + 1

    count_results = np.array(count_results)
    index_to_discard = np.where(count_results < min_count)

    boxes = np.delete(boxes, index_to_discard, 0)

    #return count_results
    return [list(box) for box in boxes]


def combine_boxes_intersect(boxes):
    if not boxes:
        return []
    noIntersectLoop = False
    noIntersectMain = False
    posIndex = 0
    # keep looping until we have completed a full pass over each rectangle
    # and checked it does not overlap with any other rectangle
    while noIntersectMain == False:
        noIntersectMain = True
        posIndex = 0
        # start with the first rectangle in the list, once the first
        # rectangle has been unioned with every other rectangle,
        # repeat for the second until done
        while posIndex < len(boxes):
            noIntersectLoop = False
            while noIntersectLoop == False and len(boxes) > 1:
                a = boxes[posIndex]
                listBoxes = np.delete(boxes, posIndex, 0)
                index = 0
                for b in listBoxes:
                    #if there is an intersection, the boxes overlap
                    if intersection(a, b):
                        #newBox = union(a,b)
                        newBox = intersection(a, b)
                        listBoxes[index] = newBox
                        boxes = listBoxes
                        noIntersectLoop = False
                        noIntersectMain = False
                        index = index + 1
                        break
                    noIntersectLoop = True
                    index = index + 1
            posIndex = posIndex + 1


    return boxes

def filter_boxes_by_radiologist_consensus(image_id, annotations, min_consensus = 3):

    # boxes per image_id
    image_boxes = annotations.loc[annotations['image_id'] == image_id].reset_index(drop = True)

    consensus_boxes = []
    consensus_class_ids = []


    for class_name in image_boxes['class_name'].unique():

        image_boxes_class = image_boxes.loc[image_boxes['class_name'] == class_name]

        tmp_boxes = np.array(image_boxes_class[['x_min', 'y_min', 'w', 'h']]).tolist()

        boxes_after_consensus = count_boxes_intersect(tmp_boxes, min_count = min_consensus)

        if len(boxes_after_consensus) > 0:
            consensus_boxes.append(boxes_after_consensus)
            consensus_class_ids.append([class_name] * len(boxes_after_consensus))

    consensus_boxes = [x for boxes in consensus_boxes for x in boxes]
    consensus_class_ids = [x for classid in consensus_class_ids for x in classid]

    #return [image_id] * len(consensus_boxes), consensus_boxes, consensus_class_ids

    # Return data frame
    ret_df = pd.DataFrame.from_records(consensus_boxes, columns = ['x_min', 'y_min', 'w', 'h'])
    ret_df['image_id'] = image_id
    ret_df['class_name'] = consensus_class_ids

    return ret_df


def box_ensembling_wbf(image_id, annotations, iou_thr = 0.5, skip_box_thr = 0.00001):

    # boxes per image_id
    image_boxes = annotations.loc[annotations['image_id'] == image_id].reset_index(drop = True)

    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []

    for rad_id in image_boxes.rad_id.unique():

        image_boxes_rad = image_boxes.loc[image_boxes['rad_id'] == rad_id]

        boxes_list.append(np.array(image_boxes_rad[['x_min_norm', 'y_min_norm', 'x_max_norm', 'y_max_norm']]).tolist())
        labels_list.append(np.array(image_boxes_rad['class_id']).tolist())

        scores_list.append([0.95] * image_boxes_rad.shape[0]) # Just assume all radiologists are 0.95 score

        weights.append(1) # all radiologists are weighted equally


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
