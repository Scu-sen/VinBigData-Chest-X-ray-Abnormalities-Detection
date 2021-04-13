import time
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import pandas as pd
import numpy as np
from numpy import random
import shutil
from ensemble_boxes import *
import cv2
import torch
import torch.backends.cudnn as cudnn
import multiprocess as mp
from tqdm import tqdm

sys.path.append('yolov5')
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


# From https://github.com/ultralytics/yolov5
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
#     if classify:
#         modelc = load_classifier(name='resnet101', n=2)  # initialize
#         modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = Path(path), '', im0s

            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print('Done. (%.3fs)' % (time.time() - t0))

def get_yolo_inference_df(yolo_pred_path, meta):

    output = os.listdir(f'{yolo_pred_path}/labels/')
    output = [x for x in output if x.split('.')[-1] == 'txt']

    yolo_results = []

    for i in output:
        yolo_dat = pd.read_csv(f'{yolo_pred_path}/labels/{i}', sep = " ", header = None)
        yolo_dat.columns = ['yolo_class', 'x_center_norm', 'y_center_norm', 'width_norm', 'height_norm', 'confidence']
        yolo_dat['image_id'] = i.split('.')[0]

        yolo_results.append(yolo_dat)

    yolo_dat = pd.concat(yolo_results)

    yolo_dat = yolo_dat.merge(meta, how = 'left', on = 'image_id')
    #yolo_dat.rename(columns = {'dim0': 'height', 'dim1': 'width'}, inplace = True)

    yolo_dat['x_min_norm'] = yolo_dat['x_center_norm'] - yolo_dat['width_norm'] / 2
    yolo_dat['x_max_norm'] = yolo_dat['x_center_norm'] + yolo_dat['width_norm'] / 2
    yolo_dat['y_min_norm'] = yolo_dat['y_center_norm'] - yolo_dat['height_norm'] / 2
    yolo_dat['y_max_norm'] = yolo_dat['y_center_norm'] + yolo_dat['height_norm'] / 2

    yolo_dat['x_min_norm'] = yolo_dat['x_min_norm'].apply(lambda x: max(x, 0))
    yolo_dat['y_min_norm'] = yolo_dat['y_min_norm'].apply(lambda x: max(x, 0))
    yolo_dat['x_max_norm'] = yolo_dat['x_max_norm'].apply(lambda x: min(x, 1))
    yolo_dat['y_max_norm'] = yolo_dat['y_max_norm'].apply(lambda x: min(x, 1))

    yolo_dat['x_min'] = (yolo_dat['x_min_norm'] * yolo_dat['width']).astype(int)
    yolo_dat['x_max'] = (yolo_dat['x_max_norm'] * yolo_dat['width']).astype(int)
    yolo_dat['y_min'] = (yolo_dat['y_min_norm'] * yolo_dat['height']).astype(int)
    yolo_dat['y_max'] = (yolo_dat['y_max_norm'] * yolo_dat['height']).astype(int)

    return yolo_dat

def wbf_ensemble(image_id, rad1, rad2, iou_thr = 0.5, skip_box_thr = 1e-10, sigma = 0.1):

    rad1 = rad1.loc[rad1['image_id'] == image_id].reset_index(drop = True)
    rad2 = rad2.loc[rad2['image_id'] == image_id].reset_index(drop = True)

    rad1_boxes = np.array(rad1[['x_min_norm', 'y_min_norm', 'x_max_norm', 'y_max_norm']]).tolist()
    rad2_boxes = np.array(rad2[['x_min_norm', 'y_min_norm', 'x_max_norm', 'y_max_norm']]).tolist()

    rad1_conf = np.array(rad1['confidence']).tolist()
    rad2_conf = np.array(rad2['confidence']).tolist()

    rad1_lab = np.array(rad1['yolo_class']).tolist()
    rad2_lab = np.array(rad2['yolo_class']).tolist()

    boxes_list = [rad1_boxes, rad2_boxes]
    scores_list = [rad1_conf, rad2_conf]
    labels_list = [rad1_lab, rad2_lab]

    weights = [1, 1]
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                                  weights = weights,
                                                  iou_thr = iou_thr,
                                                  skip_box_thr = skip_box_thr)

    ret_df = pd.DataFrame.from_records(boxes, columns = ['x_min_norm', 'y_min_norm', 'x_max_norm', 'y_max_norm'])
    ret_df['image_id'] = image_id
    ret_df['yolo_class'] = labels
    ret_df['confidence'] = scores

    return ret_df


if __name__ == '__main__':

    print('Yolov5 Inference for 512 R8 Model...')
    if os.path.isdir('tmp/R8_512'):
        shutil.rmtree(os.path.join('tmp/R8_512'))

    args = {
        'weights' : f'data/yolo/weights/best_R8_512.pt',
        'source': 'data/extracted_images/512/test/',
        'img_size': 640,
        'conf_thres': 0.00001,
        'iou_thres': 0.5,
        'device': '',
        'classes': None,
        'view_img': False,
        'save_txt': True,
        'save_conf': True,
        'agnostic_nms': False,
        'augment': True,
        'update': False,
        'project': 'tmp',
        'name': 'R8_512',
        'exist_ok': True
    }

    opt = SimpleNamespace(**args)
    detect(opt)

    print('Yolov5 Inference for 1024 R8 Model...')
    if os.path.isdir('tmp/R8_1024'):
        shutil.rmtree(os.path.join('tmp/R8_1024'))

    args = {
        'weights' : f'data/yolo/weights/best_R8_1024.pt',
        'source': 'data/extracted_images/1024/test/',
        'img_size': 1344,
        'conf_thres': 0.00001,
        'iou_thres': 0.5,
        'device': '',
        'classes': None,
        'view_img': False,
        'save_txt': True,
        'save_conf': True,
        'agnostic_nms': False,
        'augment': True,
        'update': False,
        'project': 'tmp',
        'name': 'R8_1024',
        'exist_ok': True
    }

    opt = SimpleNamespace(**args)
    detect(opt)

    metadata = pd.read_csv('data/input/test_meta.csv')
    yolo_dat_512 = get_yolo_inference_df('tmp/R8_512', metadata)
    yolo_dat_1024 = get_yolo_inference_df('tmp/R8_1024', metadata)

    all_imageids = np.unique(np.concatenate([yolo_dat_512.image_id, yolo_dat_1024.image_id]))

    pool = mp.Pool(min(mp.cpu_count(), 2))

    def wbf_ensemble_wrap(i):

        return wbf_ensemble(all_imageids[i],
                            yolo_dat_512,
                            yolo_dat_1024,
                            iou_thr=0.6)

    with pool as p:
        res = list(tqdm(p.imap(wbf_ensemble_wrap,
                               range(len(all_imageids))),
                               total = len(all_imageids)))

    pool.terminate()
    pool.join()

    yolo_dat = pd.concat(res).reset_index(drop = True)
    yolo_dat['yolo_class'] = yolo_dat['yolo_class'].astype(int)

    yolo_dat = yolo_dat.merge(metadata, how = 'left', on = 'image_id')
    #yolo_dat.rename(columns = {'dim0': 'height', 'dim1': 'width'}, inplace = True)

    yolo_dat['x_min'] = (yolo_dat['x_min_norm'] * yolo_dat['width']).astype(int)
    yolo_dat['x_max'] = (yolo_dat['x_max_norm'] * yolo_dat['width']).astype(int)
    yolo_dat['y_min'] = (yolo_dat['y_min_norm'] * yolo_dat['height']).astype(int)
    yolo_dat['y_max'] = (yolo_dat['y_max_norm'] * yolo_dat['height']).astype(int)

    # In our final solution, only class 10 is used from this detector
    yolo_dat = yolo_dat.loc[yolo_dat['yolo_class'] == 10]
    yolo_dat.to_csv('final_output/final_cls10.csv', index = False)
