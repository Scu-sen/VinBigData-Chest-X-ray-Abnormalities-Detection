import numpy as np
import pydicom
import pandas as pd
from matplotlib import patches as patches
from matplotlib import pyplot as plt
import cv2
import os
import glob
from tqdm import tqdm
import mmcv
import time


info = {"description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01"}
licenses = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
             "name": "Attribution-NonCommercial-ShareAlike License"},
            {"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 2,
             "name": "Attribution-NonCommercial License"},
            {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/", "id": 3,
             "name": "Attribution-NonCommercial-NoDerivs License"},
            {"url": "http://creativecommons.org/licenses/by/2.0/", "id": 4, "name": "Attribution License"},
            {"url": "http://creativecommons.org/licenses/by-sa/2.0/", "id": 5,
             "name": "Attribution-ShareAlike License"},
            {"url": "http://creativecommons.org/licenses/by-nd/2.0/", "id": 6,
             "name": "Attribution-NoDerivs License"},
            {"url": "http://flickr.com/commons/usage/", "id": 7, "name": "No known copyright restrictions"},
            {"url": "http://www.usa.gov/copyright.shtml", "id": 8, "name": "United States Government Work"}]
categories = [{"supercategory": "lesion", "id": 13, "name": "Pneumothorax"}]


def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1

    return " ".join(rle)


def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


def vis_SiiACR_data(csv_anno_file, data_root, vis_out_dir, show_num=1, start=0):
    """
    visualize
    """
    mmcv.mkdir_or_exist(vis_out_dir)
    df = pd.read_csv(csv_anno_file, index_col=0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    fig, ax = plt.subplots(nrows=1, ncols=show_num, sharey=True, figsize=(show_num * 10, 10))
    cur_show_num = 0
    for q, file_path in enumerate(
            glob.iglob(os.path.join(data_root, '*/*/*.dcm'))):
        if q < start:
            continue
        dataset = pydicom.dcmread(file_path)
        meta_data = dicom_to_dict(dataset, file_path, df)
        src_img = dataset.pixel_array
        img_clahe = clahe.apply(src_img)
        print('ori_data:{}  clahe_data:{}'.format(
            (src_img.min(), src_img.max()), (img_clahe.min(), img_clahe.max())))
        ax[cur_show_num].imshow(img_clahe, cmap=plt.cm.bone)
        # x-ray that have annotations
        rles = df[df['ImageId'] == dataset.SOPInstanceUID]['EncodedPixels'].values
        if rles[0] == '-1':
            continue
        if rles[0] != '-1':
            # if len(rles) > 1:
            #     print('{}\n{}'.format(rles, file_path))
            mask = rle2mask(rles[0], 1024, 1024).T
            y1, y2, x1, x2 = box_from_mask(mask)
            ax[cur_show_num].set_title('See Marker')
            ax[cur_show_num].imshow(mask, alpha=0.3, cmap="Reds")
            bbox = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax[cur_show_num].add_patch(bbox)
        else:
            ax[cur_show_num].set_title('Nothing to see')
        cur_show_num += 1
        if cur_show_num >= show_num:
            break
    plt.savefig(os.path.join(vis_out_dir, 'siimACR_vis.jpg'))
    plt.close()


def dicom_to_dict(dicom_data, file_path, rles_df, encoded_pixels=True):
    """Parse DICOM dataset and returns a dictonary with relevant fields.

    Args:
        dicom_data (dicom): chest x-ray data in dicom format.
        file_path (str): file path of the dicom data.
        rles_df (pandas.core.frame.DataFrame): Pandas dataframe of the RLE.
        encoded_pixels (bool): if True we will search for annotation.

    Returns:
        dict: contains metadata of relevant fields.
    """

    data = {}

    # Parse fields with meaningful information
    data['patient_name'] = dicom_data.PatientName
    data['patient_id'] = dicom_data.PatientID
    data['patient_age'] = int(dicom_data.PatientAge)
    data['patient_sex'] = dicom_data.PatientSex
    data['pixel_spacing'] = dicom_data.PixelSpacing
    data['file_path'] = file_path
    data['id'] = dicom_data.SOPInstanceUID

    # look for annotation if enabled (train set)
    if encoded_pixels:
        encoded_pixels_list = rles_df[rles_df['ImageId'] == dicom_data.SOPInstanceUID]['EncodedPixels'].values

        pneumothorax = False
        for encoded_pixels in encoded_pixels_list:
            if encoded_pixels != '-1':
                pneumothorax = True

        # get meaningful information (for train set)
        data['encoded_pixels_list'] = encoded_pixels_list
        data['has_pneumothorax'] = pneumothorax
        data['encoded_pixels_count'] = len(encoded_pixels_list)

    return data


def box_from_mask(mask):
    """mask (np.ndarray): (h, w)"""
    # return max and min of a mask to draw bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def bbox2mask(bbox):
    area = int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    polys = [bbox[0], bbox[1],  # tl
             bbox[2], bbox[1],
             bbox[2], bbox[3],
             bbox[0], bbox[3]
            ]
    polys = list(map(int, polys))
    return [polys], area


def preprocess_SiimACR(csv_anno_file, data_root, save_dir, save_png=False):
    """
    Read dicom data, and save it into png image.
    get bounding boxes from mask annotations
    1. Read csv annotations
    2. For each train dicom file, convert to png (optionally).
    3. get the bounding boxes annotations in coco format.
    4. save coco format annotations into json file
    """
    save_img_dir = save_dir + '/imgges'
    save_anno_dir = save_dir + '/annotations'
    mmcv.mkdir_or_exist(save_img_dir)
    mmcv.mkdir_or_exist(save_anno_dir)

    df = pd.read_csv(csv_anno_file, index_col=0)

    coco_annotations = {}
    images = []
    annotations = []

    num_imgs = 0
    num_anno = 0
    for file_path in tqdm(glob.iglob(os.path.join(data_root, '*/*/*.dcm'))):
        dicom_data = pydicom.dcmread(file_path)
        train_meta = dicom_to_dict(dicom_data, file_path, df)
        # save images
        pixel_array = dicom_data.pixel_array
        if save_png:
            mmcv.imwrite(
                pixel_array,
                file_path=os.path.join(save_img_dir, train_meta['id'] + '.png'))
        if save_png:
            filename = train_meta['id'] + '.png'
        else:
            filename = '/'.join(file_path.split('/')[-3:])
        image_info = dict(
            id=num_imgs,
            width=int(pixel_array.shape[1]),
            height=int(pixel_array.shape[0]),
            file_name=filename,
            date_captured=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            patient_name=str(train_meta['patient_name']),
            patient_id=str(train_meta['patient_id']),
            patient_age=int(train_meta['patient_age']),
            patient_sex=str(train_meta['patient_sex']),
            pixel_spacing=list(map(float, train_meta['pixel_spacing']))
        )
        images.append(image_info)

        if not train_meta['has_pneumothorax']:  # there is no mask annotation
            num_anno += 1
            anno_info = dict(
                id=num_anno,
                image_id=num_imgs,
                category_id=15,  # normal image
                iscrowd=0,
                segmentation=[],
                area=1,
                bbox=[0, 0, 1, 1])
            annotations.append(anno_info)
        else:
            # get bounding box annotations
            for anno_idx in range(train_meta['encoded_pixels_count']):
                num_anno += 1
                encoded_rle = train_meta['encoded_pixels_list'][anno_idx]
                mask = rle2mask(encoded_rle, 1024, 1024).T
                y1, y2, x1, x2 = box_from_mask(mask)
                segm, area = bbox2mask([x1, y1, x2, y2])
                anno_info = dict(
                    id=num_anno,
                    image_id=num_imgs,
                    category_id=13,  # image with lesion
                    iscrowd=0,
                    segmentation=segm,
                    area=area,
                    bbox=[x1, y1, x2 - x1, y2 - y1])
                annotations.append(anno_info)
        num_imgs += 1
        # if num_imgs >= 200:
        #     break
    coco_annotations['info'] = info
    coco_annotations['licenses'] = licenses
    coco_annotations['categories'] = categories
    coco_annotations['annotations'] = annotations
    coco_annotations['images'] = images

    print('convert done\nsaving coco annotations')
    mmcv.dump(coco_annotations, os.path.join(save_anno_dir, 'SiimACR_annos.json'))
    print('all done!')


if __name__ == '__main__':
    root_dir = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/SIIM_ACR'
    csv_anno_file = os.path.join(root_dir, 'stage_2_train.csv')
    data_root = os.path.join(root_dir, 'dicom_images_train')
    vis_out_dir = os.path.join(root_dir, 'demo')
    # 1. Visualize the dicom image
    # vis_SiiACR_data(csv_anno_file, data_root, vis_out_dir, show_num=20, start=32)
    #
    save_dir = os.path.join(root_dir, 'processed_data')
    preprocess_SiimACR(csv_anno_file, data_root, save_dir)

    # data_path = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/SIIM_ACR/dicom_images_train/' \
    #             '1.2.276.0.7230010.3.1.2.8323329.13157.1517875243.927170/' \
    #             '1.2.276.0.7230010.3.1.3.8323329.13157.1517875243.927169/' \
    #             '1.2.276.0.7230010.3.1.4.8323329.13157.1517875243.927171.dcm'
    # start = time.time()
    # for _ in range(200):
    #     data = pydicom.dcmread(data_path)
    #     pixel_array = data.pixel_array
    #     img = np.stack([pixel_array] * 3).transpose((1, 2, 0))
    # print(img.shape)
    # end = time.time()
    # print('avg_time: ', (end - start) / 200)