import argparse
import os
import platform
import shutil
import time
from pathlib import Path
from tqdm import tqdm

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import create_inference_dataloader
from utils.general import (check_img_size, non_max_suppression, apply_classifier, scale_coords,
                           xyxy2xywh, plot_one_box, strip_optimizer, bbox2result)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.vinbigdata import (result2vinbigdata, pad_lesion_image_to_sample, eval_from_csv_yolomAP,
                              results2vinbigdata_twoThr, VinBigData_class_names)


def transform_meta_info(metas):
    """metas: dict[list[Tensor]]:
    {
        ori_shape (list[Tensor]): h, w
        ratio (list[Tensor]): r_h, r_w
        pad (list[Tensor]):  p_w, p_h
    }
    Returns:
        list[dict]: each dict is the meta info each image in the batch
        [
            {
                ori_shape (tuple): (h, w)
                ratio (tuple): (h, w)
                pad (tuple): (h, w)
            }
        ]
    """
    new_metas = [dict() for _ in range(metas['ori_shape'][0].shape[0])]
    for key, val in metas.items():
        val = [v.cpu().numpy().tolist() for v in val]
        for im_i, (v1, v2) in enumerate(zip(*val)):
            new_metas[im_i][key] = (float(v1), float(v2))
    return new_metas


def detect():
    out, info_file, img_root, weights, view_img, save_txt, imgsz = \
        opt.output, opt.info_file, opt.img_root, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    device = select_device(opt.device)
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    os.makedirs(out, exist_ok=True)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataloader, dataset = create_inference_dataloader(
        info_file, img_root, imgsz, opt.batch_size, model.stride.max(), rect=opt.rect, pad=0.5)

    # Get names and colors
    try:
        names = model.module.names if hasattr(model, 'module') else model.names
    except:
        names = VinBigData_class_names
    np.random.seed(40)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    results = []
    img_names = []
    saved_im_num = 0
    for batch_i, (path, img, metas) in enumerate(tqdm(dataloader)):
        # path: (list[str]), img: Tensor (n, c, h, w), im0s(Tensor): [n, h, w, c]
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS  list[Tensor](k, 6): (x1, y1, x2, y2, score, cls)
        pred = non_max_suppression(pred, opt.save_thres, opt.iou_thres,
                                   classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        metas = transform_meta_info(metas)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, shape_info = path[i], '', metas[i]
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(shape_info['ori_shape'])[[1, 0, 1, 0]]  # normalization gain whwh
            if saved_im_num < opt.save_img_num:
                ori_img = cv2.imread(p)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0_shape=shape_info['ori_shape'],
                                          ratio_pad=(shape_info['ratio'], shape_info['pad'])
                                          if shape_info.get('ratio') is not None else None).round()

                # save results correspond to the mmdet format
                results.append(bbox2result(det[:, :-1], det[:, -1].long(), opt.num_cls))
                img_names.append(os.path.splitext(os.path.basename(p))[0])

                # Write results, loop through each detected box
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    # Add bbox to image
                    if (saved_im_num < opt.save_img_num or view_img) and (conf >= opt.vis_thres):
                        label = '{}_{:.1f}'.format(names[int(cls)], float(conf) * 100)
                        plot_one_box(xyxy, ori_img, label=label, color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, ori_img)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if saved_im_num < opt.save_img_num:
                cv2.imwrite(save_path, ori_img)
                saved_im_num += 1
        # if batch_i >= 5:
        #     break
    # inference all images done, save testing results
    out_csv_file = os.path.join(out, 'results_{}_cls{}.csv'.format(opt.split, opt.post_fix))
    result2vinbigdata(results, out_csv_file, img_names,
                      pre_scale=1.0, score_thr=opt.save_thres)
    if opt.high_thr is not None:  # two-threshold method for post-processing
        two_thr_out_file = os.path.join(out, 'results_{}_cls_twoThr.csv'.format(opt.split))
        results2vinbigdata_twoThr(out_csv_file, two_thr_out_file, high_thr=opt.high_thr, low_thr=opt.save_thres)

    if opt.split == 'test':
        pad_lesion_image_to_sample(
            res_det_csv=out_csv_file,
            sub_sample_csv=opt.sample_sub_file,
            out_file=out_csv_file)
        if opt.high_thr is not None:
            pad_lesion_image_to_sample(
                res_det_csv=two_thr_out_file,
                sub_sample_csv=opt.sample_sub_file,
                out_file=two_thr_out_file)
    elif opt.split in ['train', 'val']:
        eval_from_csv_yolomAP(out_csv_file, dataset.anno_file,
                              num_classes=15 if opt.high_thr is not None else 14,
                              KeepNoAnnoImgs=opt.high_thr is not None)
        if opt.high_thr is not None:
            print('after two-the method')
            # to keep up with the evaluation method online, also evaluate the Nofinding class.
            eval_from_csv_yolomAP(two_thr_out_file, dataset.anno_file,
                                  num_classes=15 if opt.high_thr is not None else 14,
                                  KeepNoAnnoImgs=True)

    if save_txt or saved_im_num > 0:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='model.pt path(s)')
    parser.add_argument('--info_file', type=str, default='inference/test.json', help='json file for the testing image list')
    parser.add_argument('--img_root', type=str, default='inference/images', help='image root for testing images')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--save_thres', type=float, default=0.01, help='object confidence threshold for file saving')
    parser.add_argument('--vis_thres', type=float, default=0.3, help='object confidence threshold for visualization')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--batch_size', type=int, default=1, help='inference batch size.')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--num_cls', type=int, default=15, help='number of classes, nc + 1')
    parser.add_argument('--sample_sub_file', type=str, default=None, help='path to load sample submission file.')
    parser.add_argument('--save_img_num', type=int, default=-1, help='number of saving image for visualization.')
    parser.add_argument('--rect', action='store_true', help='testing with rect images for batch testing.')
    parser.add_argument('--split', type=str, default='test', help='testing with rect images for batch testing.')
    parser.add_argument('--cls_offset', type=int, default=0, help="for 2-class detector, revise the cls_id")
    parser.add_argument('--high_thr', type=float, default=None, help="The high score for two-threshold method")
    parser.add_argument('--post_fix', type=str, default='', help='the post_fix for saving csv file.')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
