import cv2
import os
import mmcv
from tqdm import tqdm
from collections import defaultdict


def get_cls_image_patch(json_anno_file, src_dir, save_dir):
    """
    Extract image patches for each of the classes in image.
    output file structure:
    save_dir:
    |--cls0
    |--|--img1.png
    |--cls1
    |--|--img1.png
    ...
    """
    coco_annotations = mmcv.load(json_anno_file)
    image_bboxes = defaultdict(list)

    for anno in coco_annotations['annotations']:
        image_bboxes[anno['image_id']].append(anno)

    image_infos = {info['id']: info for info in coco_annotations['images']}

    save_info = defaultdict(list)
    save_cls_num = defaultdict(int)
    for im_id, im_bboxes in tqdm(image_bboxes.items(), total=len(image_bboxes)):
        filename = image_infos[im_id]['file_name']
        image = cv2.imread(os.path.join(src_dir, filename))

        for boxes in im_bboxes:
            x1, y1, w, h = boxes['bbox']
            x2, y2 = x1 + w, y1 + h
            cat_id = boxes['category_id']
            if cat_id == 15:
                continue

            img_patch = image[y1:y2, x1:x2]

            out_file = os.path.join(save_dir, 'cls_{}/{}_{}.png'.format(
                cat_id, os.path.splitext(filename)[0], save_cls_num[cat_id]))
            mmcv.imwrite(img_patch, out_file)

            save_info[cat_id].append(
                dict(box=(x1, y1, x2, y2),
                     cat_id=cat_id,
                     filename='cls_{}/{}_{}.png'.format(cat_id, os.path.splitext(filename)[0], save_cls_num[cat_id])))
            save_cls_num[cat_id] += 1
    mmcv.dump(save_info, os.path.join(save_dir, 'meta_info.pkl'))
    print('all done!')


if __name__ == '__main__':
    json_anno_file = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/vinbig_train.json'
    src_dir = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/train'
    save_dir = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/cls_image_patch'
    get_cls_image_patch(json_anno_file, src_dir, save_dir)
