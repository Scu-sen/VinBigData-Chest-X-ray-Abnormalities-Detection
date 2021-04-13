import mmcv
import os
from tqdm import tqdm


def merge_coco_annotations(anno_file_list, save_dir, pre_fixs, keep_cls=13):
    """merge multi coco annotations into one for training 2-class object detection model.
    Args:
        anno_file_list (list[str]): list of files to load coco json annotations
        save_dir (str): path to save coco json annotations
        pre_fixs (list[str]): list of prefix that add to the 'file_name' field
        keep_cls (int): keep the classes we need only, exclude other classes.
    """
    annos = mmcv.load(anno_file_list[0])
    categories =[info for info in annos['categories'] if info['id'] == keep_cls]
    info = annos['info']
    licenses = annos['licenses']
    del annos

    new_coco_annotations = {}
    new_annotations = []
    new_images = []
    image_idx_offset = 0
    anno_idx_offset = 0
    for file_i, file_path in tqdm(enumerate(anno_file_list), total=len(anno_file_list)):
        annos = mmcv.load(file_path)
        cur_images = annos['images']
        cur_annotations = annos['annotations']
        for img_info in cur_images:  # loop through image_infos
            img_info['id'] += image_idx_offset
            img_info['file_name'] = os.path.join(pre_fixs[file_i], img_info['file_name'])
            new_images.append(img_info)
        for anno_info in cur_annotations:  # loop through boxes
            anno_info['id'] += anno_idx_offset
            anno_info['image_id'] += image_idx_offset
            cat_id = keep_cls if anno_info['category_id'] == keep_cls else 15
            anno_info['category_id'] = cat_id
            new_annotations.append(anno_info)

        image_idx_offset += len(cur_images)
        anno_idx_offset += len(cur_annotations)

    new_coco_annotations['info'] = info
    new_coco_annotations['licenses'] = licenses
    new_coco_annotations['categories'] = categories
    new_coco_annotations['annotations'] = new_annotations
    new_coco_annotations['images'] = new_images

    print('convert done\nsaving coco annotations')
    mmcv.dump(new_coco_annotations, save_dir)
    print('all done!')


if __name__ == '__main__':
    # process training annotations
    # anno_file_list = ['/mnt/group-ai-medical-2/private/zehuigong/dataset1/SIIM_ACR/processed_data/'
    #                   'annotations/SiimACR_annos.json',
    #                   '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/'
    #                   'processed_data/new_Kfold_annotations/rad_id8/vinbig_train_fold3.json']
    # save_dir = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/SIIM_ACR/processed_data/annotations'
    # pre_fixs = ['siim_acr/images/dicom_images_train',
    #             'vinbigdata/images/train']
    # merge_coco_annotations(anno_file_list, save_dir, pre_fixs, keep_cls=13)

    # process validation annotations
    val_anno_file_list = ['/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/'
                          'processed_data/new_Kfold_annotations/rad_id8/vinbig_val_fold3.json']
    save_dir = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/SIIM_ACR/processed_data/annotations/mergedAnnos_val_rad8F3_cls13.json'
    pre_fixs = ['', ]
    merge_coco_annotations(val_anno_file_list, save_dir, pre_fixs, keep_cls=13)