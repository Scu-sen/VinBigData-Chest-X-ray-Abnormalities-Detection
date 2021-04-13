import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread
import pydicom
import mmcv
from collections import defaultdict
import operator

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from pycocotools.coco import COCO
import albumentations as A

# import sys
# sys.path.insert(0, './')
from utils import comm
from utils.general import xyxy2xywh, xywh2xyxy, torch_distributed_zero_first, labels_to_class_weights, plot_one_box

help_url = ''
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']
cls_img_patch_path = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/cls_image_patch'
lesion_location_file = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/lesion_location/class_lesion_boxes.pkl'

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def read_img(file_path, self=None, need_paste=False, index=None):
    """
    Read jpg png or dicom image.
    if needed, use cls image patch to paste on the specific region of image.
    if needed, crop the specific region of image. (For some special cases, e.g., AAFMA dataset)
    """
    # read image
    ext = os.path.splitext(file_path)[1]
    if ext in img_formats:
        img0 = cv2.imread(file_path)  # BGR
    elif ext in ['.dcm', '.dicom']:
        data = pydicom.dcmread(file_path)
        img0 = data.pixel_array
        img0 = np.stack([img0] * 3).transpose((1, 2, 0))
    elif ext in ['.npy', '.npz']:
        data = np.load(file_path)
        img0 = np.stack([data] * 3).transpose((1, 2, 0))
    else:
        raise ValueError('unexpected img format: {} file_path:{}'.format(ext, file_path))
    assert img0 is not None, 'Image Not Found ' + file_path

    # normalize image (optional)
    if getattr(self, 'with_mean_std', False):
        img0 = (img0 - self.img_means[index]) / self.img_stds[index]
        img0 = (img0 - img0.min()) / (img0.max() - img0.min())  # normalize to 0~1
        img0 = (img0 * 255).astype(np.uint8)

    # crop image (optional)
    if getattr(self, 'crop_region', None) is not None:
        im_h, im_w = img0.shape[:2]
        cr = np.array(self.crop_region, dtype=np.float32)
        cr[0:4:2] *= im_w
        cr[1:4:2] *= im_h
        cr = np.round(cr).astype(np.int)
        img0 = img0[cr[1]: cr[3], cr[0]: cr[2], :]

    # sample cls image patch and paste (optional)
    if need_paste:
        nc = 14
        total_num = random.randint(1, 5)  # sample total object in one image
        chosen_classes = np.random.choice(nc, size=total_num, replace=True, p=self.class_weights)
        cls_num = np.bincount(chosen_classes, minlength=nc)
        print('chosen classes:{}\ncls_num:{}'.format(chosen_classes, cls_num))
        labels = []
        im_h, im_w = img0.shape[:2]
        for cls, num in enumerate(cls_num):
            if num < 1:
                continue
            chosen_patches = random.choices(self.cls_image_patch_info[cls], k=num)
            for patch in chosen_patches:
                x1, y1, x2, y2 = patch['box']
                w, h = x2 - x1, y2 - y1
                cat_id = patch['cat_id'] - 1
                img_file = os.path.join(cls_img_patch_path, patch['filename'])
                img_patch = cv2.imread(img_file)
                assert img_patch is not None, img_file
                rx1, ry1, rx2, ry2 = random.choice(self.lesion_location[cat_id])[:4] * [im_w, im_h, im_w, im_h]
                ct_x = np.random.randint(rx1, rx2, size=1)[0]
                ct_y = np.random.randint(ry1, ry2, size=1)[0]
                x1, x2 = round(ct_x - w * 0.5), round(ct_x + w * 0.5)
                y1, y2 = round(ct_y - h * 0.5), round(ct_y + h * 0.5)
                # correct coordinates
                if x1 < 0:
                    left, x1 = -x1, 0
                    x2 += left
                if y1 < 0:
                    up, y1 = -y1, 0
                    y2 += up
                if x2 >= im_w:
                    right, x2 = x2 - im_w + 1, im_w - 1
                    x1 -= right
                if y2 >= im_h:
                    bottom, y2 = y2 - im_h + 1, im_h - 1
                    y1 -= bottom
                print(x1, y1, x2, y2)
                img0[y1:y2, x1:x2] = img_patch
                labels.append([cat_id, (x1 + x2) / (2 * im_w), (y1 + y2) / (2 * im_h),
                               (x2 - x1) / im_w, (y2 - y1) / im_h])
        labels = np.array(labels, dtype=np.float32)
        return img0, labels

    return img0


def read_neigh_imgs(file_path, self):
    """
    Read the neighboring images centered with the current image, to consider the relation information
    exist in the depth direction, which is so-called the 2.5D.
    """
    ext = os.path.splitext(file_path)[1]
    dir_path = os.path.dirname(file_path)
    def make_patch_complete(start, end, info):
        if start < info[0]:
            end = info[0] + end - start
            start = info[0]
        if end > info[1]:
            start = info[1] - (end - start)
            end = info[1]
        return start, end

    # get the indexes of the neighboring slice
    num_neig_imgs = self.num_neig_imgs  # e.g. 1, 2, 3
    pat_id, cur_slice_idx = file_path.split('/')[-2:]
    cur_slice_idx = int(os.path.splitext(cur_slice_idx)[0])
    pat_slice_info = self.pat_slice_infos[pat_id]
    start = cur_slice_idx - num_neig_imgs
    end = cur_slice_idx + num_neig_imgs
    start, end = make_patch_complete(start, end, pat_slice_info)

    # read the neighboring slices
    imgs = []
    for idx in range(start, end + 1):
        file = os.path.join(dir_path, '{}{}'.format(idx, ext))
        cur_img = read_img(file, self=self)
        imgs.append(cur_img[:, :, 1:2])
    imgs = np.concatenate(imgs, axis=2)
    return imgs


def create_dataloader(ann_file, img_root, imgsz, batch_size, stride, opt, hyp=None, augment=False,
                      cache=False, pad=0.0, rect=False, local_rank=-1, world_size=1, image_weights=False,
                      keep_no_anno_img=False, split='train', ratio=1.0):
    dataset_cls = LoadImagesAndLabelsKeepNormal if keep_no_anno_img else LoadImagesAndLabels
    kwargs = dict(split=split, ratio=ratio) if keep_no_anno_img else dict()
    # dataset_cls = LoadImagesAndLabelsSpeCls
    # kwargs = dict()
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    with torch_distributed_zero_first(local_rank):
        dataset = dataset_cls(ann_file=ann_file,
                              img_root=img_root,
                              img_size=imgsz,
                              batch_size=batch_size,
                              augment=augment,  # augment images
                              hyp=hyp,  # augmentation hyperparameters
                              rect=rect,  # rectangular training
                              image_weights=image_weights,
                              cache_images=cache,
                              single_cls=opt.single_cls,
                              stride=int(stride),
                              pad=pad,
                              **kwargs)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if local_rank != -1 else None
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             sampler=train_sampler,
                                             pin_memory=True,
                                             collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


def create_inference_dataloader(test_imgs_file, img_root, imgsz, batch_size, stride, rect=False,
                                pad=0.0, local_rank=-1, world_size=1, crop_region=None, collect_pat_slice=False,
                                num_neig_imgs=-1):
    with torch_distributed_zero_first(local_rank):
        dataset = LoadImagesFromInfoFile(test_imgs_file, img_root, imgsz,
                                         batch_size=batch_size,
                                         rect=rect,
                                         stride=int(stride),
                                         pad=pad,
                                         crop_region=crop_region,
                                         collect_pat_slice=collect_pat_slice,
                                         num_neig_imgs=num_neig_imgs)
    batch_size = min(batch_size, len(dataset))
    if rect:  # batch testing
        nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, 8])  # number of workers
    else:
        nw = 1
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=False,
                                             pin_memory=True)
    return dataloader, dataset


class LoadImages:  # for inference
    def __init__(self, path, img_size=640):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (p, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nf, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nf, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe=0, img_size=640):
        self.img_size = img_size

        if pipe == '0':
            pipe = 0  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa'  # IP traffic camera
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        # https://answers.opencv.org/question/215996/changing-gstreamer-pipeline-to-opencv-in-pythonsolved/
        # pipe = '"rtspsrc location="rtsp://username:password@192.168.1.64/1" latency=10 ! appsink'  # GStreamer

        # https://answers.opencv.org/question/200787/video-acceleration-gstremer-pipeline-in-videocapture/
        # https://stackoverflow.com/questions/54095699/install-gstreamer-support-for-opencv-python-package  # install help
        # pipe = "rtspsrc location=rtsp://root:root@192.168.0.91:554/axis-media/media.amp?videocodec=h264&resolution=3840x2160 protocols=GST_RTSP_LOWER_TRANS_TCP ! rtph264depay ! queue ! vaapih264dec ! videoconvert ! appsink"  # GStreamer

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, 'Camera Error %s' % self.pipe
        img_path = 'webcam.jpg'
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640):
        self.mode = 'images'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(0 if s == '0' else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def collect_imgs_PatWise(self):
    """collect images that belong to one patient together
    Return:
        {
            pat_id: list[int], (start, end, num_slice) the start and end slice index of the patient
                    and also the number of total slice.
        }
    """
    image_files = self.img_files
    pat_img_files = defaultdict(list)
    for file in image_files:
        pat_id, slice = file.split('/')[-2:]
        pat_img_files[pat_id].append(int(os.path.splitext(slice)[0]))
    for pat_id in list(pat_img_files.keys()):
        slice_idxes = np.array(pat_img_files[pat_id], dtype=np.int)
        slice_idxes = np.sort(slice_idxes)
        pat_img_files[pat_id] = [slice_idxes[0], slice_idxes[-1], slice_idxes.shape[0]]
    return pat_img_files


class LoadImagesFromInfoFile(Dataset):
    def __init__(self, anno_file, img_root, img_size=640, batch_size=1, rect=False,
                 stride=32, pad=0.0, crop_region=None, collect_pat_slice=False, num_neig_imgs=-1):
        self.anno_file = anno_file
        self.img_size = img_size
        self.img_root = img_root
        self.rect = rect
        self.stride = stride
        self.batch_size = batch_size

        # parameters that specific for aafma dataset
        self.crop_region = crop_region
        self.num_neig_imgs = num_neig_imgs

        self.img_files, self.shapes, img_means, img_stds = self.load_image_files(anno_file)
        self.with_mean_std = False
        if len(img_means) > 10:
            self.with_mean_std = True
            self.img_means = np.array(img_means)
            self.img_stds = np.array(img_stds)

        if collect_pat_slice:
            self.resort_image_files()
        if self.num_neig_imgs > 0:
            self.pat_slice_infos = collect_imgs_PatWise(self)
        bi = np.floor(np.arange(len(self.img_files)) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi

        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            if self.with_mean_std:
                self.img_means = self.img_means[irect]
                self.img_stds = self.img_stds[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

    def load_image_files(self, anno_file):
        img_form = os.path.splitext(os.listdir(self.img_root)[0])[1]
        coco = COCO(anno_file)
        img_ids = coco.getImgIds()
        img_files = []
        im_shapes = []  # (w, h)
        means = []
        stds = []
        for i in img_ids:  # loop through images
            info = coco.loadImgs([i])[0]
            if self.crop_region is None:
                im_shapes.append([info['width'], info['height']])
            else:  # cropping testing, need to update the img shape info.
                ori_im_w, ori_im_h = info['width'], info['height']
                cx1, cy1, cx2, cy2 = self.crop_region
                new_w = (cx2 - cx1) * ori_im_w
                new_h = (cy2 - cy1) * ori_im_h
                im_shapes.append([new_w, new_h])
            img_files.append(os.path.join(self.img_root,
                                          os.path.splitext(info['file_name'])[0] + img_form if img_form != '' else
                                          info['file_name']))
            # add normalized info
            if 'mean' in info:
                means.append(info['mean'])
                stds.append(info['std'])
        return img_files, np.array(im_shapes), means, stds

    def resort_image_files(self):
        """collect image_files patient wise"""
        image_files = self.img_files
        pat_image_files = defaultdict(list)
        for idx, file in enumerate(image_files):
            pat_image_files[file.split('/')[-2]].append((file, idx))
        new_image_files = []
        new_im_shapes = []
        for pat_id, pat_files in pat_image_files.items():  # loop through pats
            for (file, i) in pat_files:  # loop through slice in one pat
                new_image_files.append(file)
                new_im_shapes.append(self.shapes[i])
        self.img_files = new_image_files
        self.shapes = np.array(new_im_shapes)

    def __len__(self):
        return len(self.img_files)

    def load_rect_image(self, index):
        path = self.img_files[index]
        if self.num_neig_imgs > 0:  # cat multiple slices together along channel dimension
            img0 = read_neigh_imgs(path, self=self)
        else:
            img0 = read_img(path, self=self, index=index)

        h0, w0 = img0.shape[:2]
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
        else:
            img = img0
        h, w = img.shape[:2]
        # final letterboxed shape
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=False)

        # convert  BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = (img / 255.).astype(np.float32)

        shapes = dict(ori_shape=(h0, w0),
                      ratio=(h / h0 * ratio[1], w / w0 * ratio[0]),
                      pad=pad)
        return path, img, shapes

    def load_rect_image_normalize(self, index):
        path = self.img_files[index]
        assert path.endswith('.npy')
        if self.num_neig_imgs > 0:  # cat multiple slices together along channel dimension
            img0 = read_neigh_imgs(path, self=self)
        else:
            img0 = read_img(path, self=self, index=index)

        img0 = (img0 - self.img_means[index]) / self.img_stds[index]

        h0, w0 = img0.shape[:2]
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
        else:
            img = img0
        h, w = img.shape[:2]

        # final letterboxed shape
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
        img, ratio, pad = letterbox(img, shape, color=(0, 0, 0), auto=False, scaleup=False)

        # convert  BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)

        shapes = dict(ori_shape=(h0, w0),
                      ratio=(h / h0 * ratio[1], w / w0 * ratio[0]),
                      pad=pad)
        return path, img, shapes

    def load_letterbox_img(self, index):
        path = self.img_files[index]
        if self.num_neig_imgs > 0:  # cat multiple slices together along channel dimension
            img0 = read_neigh_imgs(path, self=self)
        else:
            img0 = read_img(path, self=self, index=index)

        h0, w0 = img0.shape[:2]

        # padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = (img / 255.).astype(np.float32)

        shapes = dict(ori_shape=(h0, w0))
        return path, img, shapes

    def __getitem__(self, index):
        if self.rect:  # rect image for batch testing and be consistent with test.py
            return self.load_rect_image(index)
        else:  # single image inference
            return self.load_letterbox_img(index)


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, ann_file, img_root, img_size=640, batch_size=16, augment=False, hyp=None, rect=False,
                 image_weights=False, cache_images=False, single_cls=False, stride=32, pad=0.0, **kwargs):
        # load image files, labels, and img_shapes (w, h)
        self.img_files, self.labels, self.shapes, img_means, img_stds = \
            self.load_images_labels(ann_file, img_root)
        self.with_mean_std = False
        if len(img_means) > 1:
            self.with_mean_std = True
            self.img_means = np.array(img_means)
            self.img_stds = np.array(img_stds)
            assert len(img_means) == len(self.img_files)
        n = len(self.img_files)
        assert n > 0, 'No images found in %s. See %s' % (ann_file, help_url)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and hyp['mosaic'] and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.indices = list(range(self.n))

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            if self.with_mean_std:
                self.img_means = self.img_means[irect]
                self.img_stds = self.img_stds[irect]
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache labels
        create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        pbar = tqdm(self.labels)
        for i, l in enumerate(pbar):
            if l.shape[0]:
                # (cls, cx, cy, w, h)
                assert l.shape[1] == 5, '> 5 label columns: %s'
                assert (l >= 0).all(), 'negative labels: %s'
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels:' \
                                              ' {}\nfile:{}\n(w, h):{}\nann_file:{}'.format(l, self.img_files[i], self.shapes[i], ann_file)
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found

                # Create subdataset (a smaller dataset)
                if create_datasubset and ns < 1E4:
                    if ns == 0:
                        create_folder(path='./datasubset')
                        os.makedirs('./datasubset/images')
                    exclude_classes = 43
                    if exclude_classes not in l[:, 0]:
                        ns += 1
                        # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                        with open('./datasubset/images.txt', 'a') as f:
                            f.write(self.img_files[i] + '\n')

                # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)  # make new output folder

                        b = x[1:] * [w, h, w, h]  # box
                        b[2:] = b[2:].max()  # rectangle to square
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
            else:
                ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

            pbar.desc = 'Scanning labels (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                nf, nm, ne, nd, n)
        if nf == 0 and not hasattr(self, 'keep_no_label'):
            s = 'WARNING: No labels found in %s. See %s' % (os.sep, help_url)
            print(s)
            assert not augment, '%s. Can not train without labels.' % s

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

        # albuaugmentation
        # if self.augment:
        #     self.albu_transform =  A.Compose([
        #         A.OneOf([
        #             A.RandomGamma(gamma_limit=(60, 120), p=0.9),
        #             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        #             # A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
        #         ]),
        #         A.OneOf([
        #             A.Blur(blur_limit=4, p=1),
        #             A.MotionBlur(blur_limit=4, p=1),
        #             A.MedianBlur(blur_limit=3, p=1)
        #         ], p=0.2),
        #         A.GaussNoise(p=0.3),
        #         ])
        self.class_weights = labels_to_class_weights(self.labels, nc=14).numpy()
        self.cls_image_patch_info = mmcv.load(os.path.join(cls_img_patch_path, 'meta_info.pkl'))
        self.lesion_location = mmcv.load(lesion_location_file)

    def load_images_labels(self, ann_file, img_root):
        img_form = os.path.splitext(os.listdir(img_root)[0])[1]
        coco = COCO(ann_file)
        cat_ids = coco.getCatIds()
        cat2label = {cat_id: i
            for i, cat_id in enumerate(cat_ids)}
        self.label2cat = {l: c for (c, l) in cat2label.items()}
        img_ids = coco.getImgIds()
        image_files = []
        labels = []
        shapes = []
        means = []
        stds = []
        for i in img_ids:  # loop through images
            info = coco.loadImgs([i])[0]
            img_h = info['height']
            img_w = info['width']

            img_id = info['id']
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            ann_info = coco.loadAnns(ann_ids)
            cur_label = []
            for ann in ann_info:   # loop through objects in images
                if ann.get('ignore', False):
                    continue
                x1, y1, bw, bh = ann['bbox']
                if ann['area'] <= 0 or bw < 1 or bh < 1:
                    continue
                if ann['category_id'] == 15:  # filter images that have no annotations
                    continue
                ct_x = (x1 + bw * 0.5) / img_w
                ct_y = (y1 + bh * 0.5) / img_h
                cls_id = cat2label[ann['category_id']]
                box = [cls_id, ct_x, ct_y, bw / img_w, bh / img_h]
                cur_label.append(box)
            if cur_label:  # images with annos
                image_files.append(os.path.join(
                    img_root,
                    os.path.splitext(info['file_name'])[0] + img_form if img_form != '' else info['file_name']))
                labels.append(np.array(cur_label, dtype=np.float32))
                shapes.append([img_w, img_h])
                if 'mean' in info:  # there is mean and std in the annotation
                    means.append(info['mean'])
                    stds.append(info['std'])
        shapes = np.array(shapes)
        return image_files, labels, shapes, means, stds

    def cache_labels(self, path='labels.cache'):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()  # PIL verify
                # _ = io.imread(img)  # skimage verify (from skimage import io)
                shape = exif_size(image)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                x[img] = None
                print('WARNING: %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def load_image_and_labels(self, index):
        img, (h0, w0), (h, w) = load_image(self, index)
        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # Load labels
        labels = []
        x = self.labels[index]
        if x.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = x.copy()
            labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
        return img, labels, shapes

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(img.dtype)
                labels = np.concatenate((labels, labels2), 0)

        else:
            img, labels, shapes = self.load_image_and_labels(index)

        # self.vis_training_imgs(img, labels, index)

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
            # apply albu augmentations
            # img = self.albu_transform(image=img)["image"]
            # img, bboxes, clses = transformed['image'], transformed['bboxes'], transformed['labels']
            # labels = np.hstack([clses[:, np.newaxis], bboxes])

            # Augment colorspace
            # augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.5:  # 0.9
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = (img / 255.).astype(np.float32)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    # def vis_training_imgs(self, img, labels, index):
    #     if len(labels) < 1:
    #         return
    #     img = img.astype(np.uint8)
    #     h, w, c = img.shape
    #     if c > 3:
    #         ct = c // 2
    #         img = img[:, :, ct-1:ct+2]
    #         assert img.shape[2] == 3
    #     print('num_labels:{}'.format(labels.shape[0]))
    #     # new_labels = np.zeros(labels.shape, dtype=np.float32)
    #     # new_labels[:, 0] = labels[:, 0]
    #     # new_labels[:, 1:3] = labels[:, 1:3] - labels[:, 3:5] * 0.5
    #     # new_labels[:, 3:5] = labels[:, 1:3] + labels[:, 3:5] * 0.5
    #     # new_labels[:, 1::2] *= w
    #     # new_labels[:, 2::2] *= h
    #     # for idx in range(labels.shape[0]):
    #     #     cls_id, xyxy = int(labels[idx, 0]), labels[idx, 1:]
    #     #     print('xyxy: ', xyxy)
    #     #     plot_one_box(xyxy.astype(np.int), img, color=(0, 0, 255),
    #     #                  label='cls_{}'.format(cls_id), line_thickness=2)
    #     print('labels: ', labels)
    #     img = draw_bbox(img, bboxes=labels[:, 1:].astype(np.int), labels=labels[:, 0])
    #     print('saving training vis img index: ', index)
    #     img_file = self.img_files[index]
    #     file_name = '_'.join(img_file.split('/')[-2:])
    #     mmcv.imwrite(img, os.path.join('/mnt/group-ai-medical-2/private/zehuigong/dataset1/A_AFMA_Detection/demo',
    #                                    'train_vis_{}'.format(file_name)))

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


class LoadImagesAndLabelsKeepNormal(LoadImagesAndLabels):
    """Training with images that have annotations and have not annotations.
    ratio (float): ratio=#normal / #abnormal, initialize a small ratio and, as the training
                   continues, increase the ratio gradually.
    """

    def __init__(self, ratio=1.0, split='train', **kwargs):
        assert split in ['train', 'val']
        spe_classes = kwargs['hyp'].get('spe_classes', None)
        assert spe_classes is None or isinstance(spe_classes, (list, tuple)), \
            'spe_classes must be tuple or list, but got {}'.format(type(spe_classes))
        self.spe_classes = spe_classes
        self.ratio = ratio
        self.split = split
        self.anno_file = kwargs['ann_file']
        self.img_root = kwargs['img_root']
        self.batch_size = kwargs['batch_size']
        self.pad = kwargs.get('pad', 0)

        # parameters that are specific for AAFMA dataset
        self.crop_region = kwargs['hyp'].get('crop_region', None)
        # number of neighboring images to concatenate, to form one unified image.
        self.num_neig_imgs = kwargs['hyp'].get('num_neig_imgs', -1)

        self.normal_imgIds, self.abnormal_imgIds = self.get_img_ids(kwargs['ann_file'])
        super(LoadImagesAndLabelsKeepNormal, self).__init__(**kwargs)

        if self.num_neig_imgs > 0:
            self.pat_slice_infos = collect_imgs_PatWise(self)

    def get_img_ids(self, ann_file):
        """get normal and abnormal img_ids separately"""
        coco_annotations = mmcv.load(ann_file)
        img_labels = defaultdict(list)
        for ann_info in coco_annotations['annotations']:
            im_id = ann_info['image_id']
            img_labels[im_id].append(ann_info['category_id'])
        normal_imgIds = []
        abnormal_imgIds = []
        for im_id, labels in img_labels.items():
            if len(labels) == 1 and labels[0] == 15:  # normal image (No finding)
                normal_imgIds.append(im_id)
            else:
                abnormal_imgIds.append(im_id)
        print('{} normal imgIds,\t{} abnormal imgIds'.format(len(normal_imgIds), len(abnormal_imgIds)))
        return normal_imgIds, abnormal_imgIds

    def load_images_labels(self, ann_file, img_root):
        """dataset for training images that have both annotations and no annotations.
           For those of the images without any annotation, sample some normal images to train.
        """
        img_form = os.path.splitext(os.listdir(img_root)[0])[1]
        coco = COCO(ann_file)
        if self.spe_classes is None:
            cat_ids = coco.getCatIds()
            cat2label = {cat_id: i
                         for i, cat_id in enumerate(cat_ids)}
        else:  # train specific class
            cat2label = {catid: idx for idx, catid in enumerate(self.spe_classes)}
        self.label2cat = {l: c for (c, l) in cat2label.items()}

        normal_imgIds = self.normal_imgIds
        abnormal_imgIds = self.abnormal_imgIds

        image_files = []
        labels = []
        shapes = []
        for i in abnormal_imgIds:  # loop through images, get abnormal images annotations
            info = coco.loadImgs([i])[0]
            img_h = info['height']
            img_w = info['width']

            if self.crop_region is not None:  # correct img shapes
                cr_img_w = int(round((self.crop_region[2] - self.crop_region[0]) * img_w))
                cr_img_h = int(round((self.crop_region[3] - self.crop_region[1]) * img_h))

            img_id = info['id']
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            ann_info = coco.loadAnns(ann_ids)
            cur_label = []
            for ann in ann_info:  # loop through objects in images
                if ann.get('ignore', False):
                    continue
                x1, y1, bw, bh = ann['bbox']
                if ann['area'] <= 0 or bw < 1 or bh < 1:
                    print('continue because area')
                    continue
                if ann['category_id'] == 15:  # filter images that have no annotations
                    continue
                if self.spe_classes is not None and ann['category_id'] not in self.spe_classes:
                    continue
                cls_id = cat2label[ann['category_id']]
                ct_x = x1 + bw * 0.5
                ct_y = y1 + bh * 0.5
                if self.crop_region is not None:  # correct annotations
                    ct_x -= self.crop_region[0] * img_w
                    ct_y -= self.crop_region[1] * img_h
                    box = [cls_id, ct_x / cr_img_w, ct_y / cr_img_h, bw / cr_img_w, bh / cr_img_h]
                else:
                    box = [cls_id, ct_x / img_w, ct_y / img_h, bw / img_w, bh / img_h]
                cur_label.append(box)
            if cur_label:  # images with annos
                image_files.append(os.path.join(
                    img_root,
                    os.path.splitext(info['file_name'])[0] + img_form if img_form != '' else info['file_name']))
                labels.append(np.array(cur_label, dtype=np.float32))
                shapes.append([img_w, img_h] if self.crop_region is None else [cr_img_w, cr_img_h])

        # sample the annotations of normal images
        if self.split == 'train':
            num_sample_normal = min(int(len(abnormal_imgIds) * self.ratio), len(normal_imgIds))
            idxes = list(range(len(normal_imgIds)))
            random.shuffle(idxes)
            sample_normal_imgIds = [normal_imgIds[idx] for idx in idxes[:num_sample_normal]]
        else:
            sample_normal_imgIds = normal_imgIds[:200]  ################[:200] change 2021-03-28######

        for i in sample_normal_imgIds:
            info = coco.loadImgs([i])[0]
            img_h = info['height']
            img_w = info['width']

            if self.crop_region is not None:  # correct img_shapes
                cr_img_w = int(round((self.crop_region[2] - self.crop_region[0]) * img_w))
                cr_img_h = int(round((self.crop_region[3] - self.crop_region[1]) * img_h))

            image_files.append(os.path.join(
                img_root,
                os.path.splitext(info['file_name'])[0] + img_form if img_form != '' else info['file_name']))
            # if self.split == 'train':  # for training, Nofinding class images have no annotations
            #     labels.append(np.zeros((0, 5), dtype=np.float32))
            # else:  # for validation, the annotations of no-finding class images are [14 0 0 1 1]
            #     labels.append(np.array([[cat2label[15], 0.5 / img_w, 0.5 / img_h, 1. / img_w, 1. / img_h]],
            #                            dtype=np.float32))
            labels.append(np.zeros((0, 5), dtype=np.float32))
            shapes.append([img_w, img_h] if self.crop_region is None else [cr_img_w, cr_img_h])

        shapes = np.array(shapes)
        return image_files, labels, shapes, [], []

    def resample_accordingTo_numObjs(self, group_ratios=(0.5, 0.4, 0.1)):
        """According to the analysis of the detection results, the performance of images that have more than
           one objects are worse than those of only one object. So sample this "hard case" more frequently to
           let model see these more.
        """
        # 1. split imgs into group according to the number of "fluid" objects
        if not hasattr(self, 'img_groups'):
            self.img_groups = self.split_img_groups()
        num_ims_group = {g: len(glist) for g, glist in self.img_groups.items()}
        max_num = max(num_ims_group.values())

        # 2. get new number of images in each group
        new_group_num = [int(len(self.labels) * gr) for gr in group_ratios]
        if sum(new_group_num) != len(self.labels):
            new_group_num[-1] = len(self.labels) - sum(new_group_num[:-1])
        assert sum(new_group_num) == len(self.labels), \
            'sum of group must be equal to the original img num! sum(group): {}  total_num:{}'.format(
                sum(new_group_num), len(self.labels))
        ori_group_idxes = [np.random.permutation(max_num)[:gn] for gn in new_group_num]
        new_indices = []
        for gi in sorted(list(self.img_groups.keys())):  # loop through each group
            new_idxes = ori_group_idxes[gi - 1] % num_ims_group[gi]
            new_indices.extend([self.img_groups[gi][idx] for idx in new_idxes])

        assert len(new_indices) == len(self.labels)

        random.shuffle(new_indices)
        return new_indices

    def split_img_groups(self):
        """
        This function is written for AAFMA dataset specifically.
        Three groups: [0, 1],  2, 3 numobjs
        """
        def get_groupId(num_obj):
            if num_obj <= 1:
                return 1
            elif num_obj == 2:
                return 2
            else:
                return 3

        group_img_list = defaultdict(list)
        for im_idx, l in enumerate(self.labels):
            idx = np.where(l[:, 0].astype(np.int) == 0)[0]  # fluid
            gId = get_groupId(idx.shape[0])
            group_img_list[gId].append(im_idx)
        return group_img_list

    def re_sample_normal_ImageIds(self, ratio=1.0):
        """
        Reset the ratio of normal images sampling
        """
        ratio = np.clip(ratio, 0, 1.0)
        self.ratio = ratio
        self.img_files, self.labels, self.shapes = self.load_images_labels(self.anno_file, self.img_root)
        self.n = len(self.img_files)
        bi = np.floor(np.arange(self.n) / self.batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi
        self.imgs = [None] * self.n

        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(np.int) * self.stride


class LoadImagesAndLabelsSpeCls(LoadImagesAndLabels):
    """
    This is the class for training the specialized detector, i.e., training on some specific classes.
    spe_classes (list[int]): the cls_id in 1-based.
    """
    def __init__(self, **kwargs):
        spe_classes = kwargs['hyp'].get('spe_classes', None)
        assert spe_classes is not None and isinstance(spe_classes, (list, tuple)), \
            'spe_classes must be tuple or list, but got {}'.format(type(spe_classes))
        self.spe_classes = spe_classes
        super(LoadImagesAndLabelsSpeCls, self).__init__(**kwargs)

    def load_images_labels(self, ann_file, img_root):
        img_form = os.path.splitext(os.listdir(img_root)[0])[1]
        coco = COCO(ann_file)
        cat2label = {catid: idx for idx, catid in enumerate(self.spe_classes)}
        self.label2cat = {l: c for (c, l) in cat2label.items()}
        img_ids = coco.getImgIds()
        image_files = []
        labels = []
        shapes = []
        for i in img_ids:  # loop through images
            info = coco.loadImgs([i])[0]
            img_h = info['height']
            img_w = info['width']

            img_id = info['id']
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            ann_info = coco.loadAnns(ann_ids)
            cur_label = []
            for ann in ann_info:   # loop through objects in images
                if ann.get('ignore', False):
                    continue
                x1, y1, bw, bh = ann['bbox']
                if ann['area'] <= 0 or bw < 1 or bh < 1:
                    continue
                if ann['category_id'] not in self.spe_classes:
                    continue
                ct_x = (x1 + bw * 0.5) / img_w
                ct_y = (y1 + bh * 0.5) / img_h
                cls_id = cat2label[ann['category_id']]
                box = [cls_id, ct_x, ct_y, bw / img_w, bh / img_h]
                cur_label.append(box)
            if cur_label:  # images with annos
                image_files.append(os.path.join(
                    img_root,
                    os.path.splitext(info['file_name'])[0] + img_form if img_form != '' else info['file_name']))
                labels.append(np.array(cur_label, dtype=np.float32))
                shapes.append([img_w, img_h])
        shapes = np.array(shapes)
        return image_files, labels, shapes, [], []


class LoadImageAndLabelsStrongWeak(LoadImagesAndLabelsKeepNormal):
    """This class is written specifically for 'unbiased teacher' paper,
    where two version of image have been produced, i.e., strong and weak augmentation"""
    def __init__(self, path_prefix='', keep_no_label=False, **kwargs):
        self.keep_no_label = keep_no_label
        self.path_prefix = path_prefix
        super(LoadImageAndLabelsStrongWeak, self).__init__(**kwargs)
        # albuaugmentation
        if self.augment:
            self.albu_transform = A.Compose([
                # A.CLAHE(clip_limit=4.0, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.8),
                A.OneOf([
                    A.Blur(blur_limit=4, p=1),
                    A.MedianBlur(blur_limit=3, p=1)
                    ], p=0.3),
                A.GaussNoise(p=0.3),
            ],
                p=1.0,
            )

    def load_images_labels(self, ann_file, img_root):
        coco = COCO(ann_file)
        cat_ids = coco.getCatIds()
        cat2label = {cat_id: i
            for i, cat_id in enumerate(cat_ids)}
        self.label2cat = {l: c for (c, l) in cat2label.items()}
        img_ids = coco.getImgIds()
        image_files = []
        labels = []
        shapes = []
        for i in img_ids:  # loop through images
            info = coco.loadImgs([i])[0]
            img_h = info['height']
            img_w = info['width']

            if self.crop_region is not None:  # correct img shapes
                cr_img_w = int(round((self.crop_region[2] - self.crop_region[0]) * img_w))
                cr_img_h = int(round((self.crop_region[3] - self.crop_region[1]) * img_h))

            img_id = info['id']
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            ann_info = coco.loadAnns(ann_ids)
            cur_label = []
            for ann in ann_info:   # loop through objects in images
                if ann.get('ignore', False):
                    continue
                x1, y1, bw, bh = ann['bbox']
                if ann['area'] <= 0 or bw < 1 or bh < 1:
                    continue
                if ann['category_id'] == 15:  # filter images that have no annotations
                    continue
                cls_id = cat2label[ann['category_id']]
                ct_x = x1 + bw * 0.5
                ct_y = y1 + bh * 0.5
                if self.crop_region is not None:  # correct annotations
                    ct_x -= self.crop_region[0] * img_w
                    ct_y -= self.crop_region[1] * img_h
                    box = [cls_id, ct_x / cr_img_w, ct_y / cr_img_h, bw / cr_img_w, bh / cr_img_h]
                else:
                    box = [cls_id, ct_x / img_w, ct_y / img_h, bw / img_w, bh / img_h]
                cur_label.append(box)
            if cur_label:  # images with annos
                image_files.append(os.path.join(img_root, self.path_prefix, info['file_name']))
                labels.append(np.array(cur_label, dtype=np.float32))
                shapes.append([img_w, img_h] if self.crop_region is None else [cr_img_w, cr_img_h])
            elif self.keep_no_label:  # images that have no annos
                image_files.append(os.path.join(img_root, self.path_prefix, info['file_name']))
                labels.append(np.zeros((0, 5), dtype=np.float32))
                shapes.append([img_w, img_h] if self.crop_region is None else [cr_img_w, cr_img_h])
        shapes = np.array(shapes)
        return image_files, labels, shapes

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        if self.mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            img, labels, shapes = self.load_image_and_labels(index)

        # self.vis_training_imgs(img, labels, index)

        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        img_strong = img.astype(np.float32)

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # apply albu augmentations, strong augmentation
            img_strong = self.albu_transform(image=img)["image"]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = (img / 255.).astype(np.float32)

        img_strong = img_strong[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_strong = np.ascontiguousarray(img_strong)
        img_strong = (img_strong / 255.).astype(np.float32)

        return torch.from_numpy(img), torch.from_numpy(img_strong), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, img_strong, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.stack(img_strong, 0), torch.cat(label, 0), path, shapes


class GroupedSemiSupDatasetTwoCrop(torch.utils.data.IterableDataset):
    """class written specifically for 'unbiased-teacher'
    !!!
       For training AAFMA dataset, keep_no_label=True for label and unlabel anno_file
       For training VinbigData, keep_no_label=False for label_ann_file,
       and keep_no_label=True for un-label_anno_file.
    !!!
    """
    def __init__(self, label_ann_file, unlabel_ann_file, **kwargs):
        self.num_workers = 1
        self.world_size = 1
        self.label_dataloader, self.label_dataset = self.create_dataset(
            ann_file=label_ann_file, keep_no_label=True, path_prefix='train', **kwargs)
        self.unlabel_dataloader, self.unlabel_dataset = self.create_dataset(
            ann_file=unlabel_ann_file, keep_no_label=True, path_prefix='test', **kwargs)
        self.batch_size = kwargs['batch_size']

    def create_dataset(self, ann_file, batch_size, stride, opt, local_rank=-1,
                       world_size=1, keep_no_label=False, **kwargs):
        with torch_distributed_zero_first(local_rank):
            dataset = LoadImageAndLabelsStrongWeak(
                ann_file=ann_file,
                batch_size=batch_size,
                single_cls=opt.single_cls,
                keep_no_label=keep_no_label,
                stride=int(stride),
                **kwargs)

        batch_size = min(batch_size, len(dataset))
        nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, 8])  # number of workers
        self.num_workers = nw
        self.world_size = world_size
        # train_sampler = comm.TrainingSampler(len(dataset))
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if local_rank != -1 else None
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=nw,
                                                 sampler=train_sampler,
                                                 pin_memory=True,
                                                 # worker_init_fn=comm.worker_init_reset_seed,
                                                 # don't batch, but yield individual elements
                                                 # collate_fn=operator.itemgetter(0)
                                                 collate_fn=LoadImageAndLabelsStrongWeak.collate_fn)
        return dataloader, dataset

    def __iter__(self):
        label_bucket, unlabel_bucket = [], []
        for d_label, d_unlabel in zip(self.label_dataloader, self.unlabel_dataloader):
            # d_label is list with len = 5, which is [img_weak, img_strong, target, file_path, shapes]
            print('I am in iter dataloader')
            if len(label_bucket) != self.batch_size:
                label_bucket.append(d_label)

            if len(unlabel_bucket) != self.batch_size:
                unlabel_bucket.append(d_unlabel)

            # yield the batch of data until all buckets are full
            if (len(label_bucket) == self.batch_size and
                len(unlabel_bucket) == self.batch_size):
                print('yieding data')
                yield (
                    LoadImageAndLabelsStrongWeak.collate_fn(label_bucket),
                    LoadImageAndLabelsStrongWeak.collate_fn(unlabel_bucket),
                )
                del label_bucket[:]
                del unlabel_bucket[:]

    def __len__(self):
        # return len(self.label_dataset) // (self.batch_size * 2 * self.world_size)
        return min(len(self.label_dataloader), len(self.unlabel_dataloader))

def dicom2array(path, voi_lut=False, fix_monochrome=True):
    """ Convert dicom file to numpy array

    Args:
        path (str): Path to the dicom file to be converted
        voi_lut (bool): Whether or not VOI LUT is available
        fix_monochrome (bool): Whether or not to apply monochrome fix

    Returns:
        Numpy array of the respective dicom file

    """
    # Use the pydicom library to read the dicom file
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    # The XRAY may look inverted
    #   - If we want to fix this we can
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data.astype(np.float32, copy=False)
    min_v = data.min()
    max_v = data.max()
    data = (data - min_v) / (max_v - min_v)

    data = data[:, :, np.newaxis]
    data = np.concatenate([data, data, data], axis=2)

    return data * 255.


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        if getattr(self, 'num_neig_imgs', -1) > 0:
            img = read_neigh_imgs(path, self)
        else:
            img = read_img(path, self=self, index=index)

        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    yc, xc = s, s  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            if img.ndim == 3:
                shape = (s * 2, s * 2, img.shape[2])
            else:
                shape = (s * 2, s * 2)
            img4 = np.full(shape, 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

        # Replicate
        # img4, labels4 = replicate(img4, labels4)

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def copyMakeBorder_torch(img, top, bottom, left, right, mode='constant', value=114):
    img = torch.from_numpy(img)
    img = torch.nn.functional.pad(img, (0, 0, left, right, top, bottom), mode=mode, value=value)
    return img.numpy()


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 128), np.mod(dh, 128)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if img.shape[2] > 3:  # cv2 cannot process img that has more than 3 channels
        img = copyMakeBorder_torch(img, top, bottom, left, right, value=color[0])
    else:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w, c = image.shape

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(c)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def reduce_img_size(path='path/images', img_size=1024):  # from utils.datasets import *; reduce_img_size()
    # creates a new ./images_reduced folder with reduced size images of maximum size img_size
    path_new = path + '_reduced'  # reduced images path
    create_folder(path_new)
    for f in tqdm(glob.glob('%s/*.*' % path)):
        try:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            r = img_size / max(h, w)  # size ratio
            if r < 1.0:
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)  # _LINEAR fastest
            fnew = f.replace(path, path_new)  # .replace(Path(f).suffix, '.jpg')
            cv2.imwrite(fnew, img)
        except:
            print('WARNING: image failure %s' % f)


def recursive_dataset2bmp(dataset='path/dataset_bmp'):  # from utils.datasets import *; recursive_dataset2bmp()
    # Converts dataset to bmp (for faster training)
    formats = [x.lower() for x in img_formats] + [x.upper() for x in img_formats]
    for a, b, files in os.walk(dataset):
        for file in tqdm(files, desc=a):
            p = a + '/' + file
            s = Path(file).suffix
            if s == '.txt':  # replace text
                with open(p, 'r') as f:
                    lines = f.read()
                for f in formats:
                    lines = lines.replace(f, '.bmp')
                with open(p, 'w') as f:
                    f.write(lines)
            elif s in formats:  # replace image
                cv2.imwrite(p.replace(s, '.bmp'), cv2.imread(p))
                if s != '.bmp':
                    os.system("rm '%s'" % p)


def imagelist2folder(path='path/images.txt'):  # from utils.datasets import *; imagelist2folder()
    # Copies all the images in a text file (list of images) into a folder
    create_folder(path[:-4])
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            os.system('cp "%s" %s' % (line, path[:-4]))
            print(line)


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


# if __name__ == '__main__':
    # class Myclass():
    #     def __init__(self):
    #         self.class_weights = np.array(
    #             [0.0092808, 0.23824, 0.069239, 0.012248, 0.11955,
    #              0.066469, 0.053303,  0.02677, 0.025763, 0.030172,
    #              0.026845, 0.013728,  0.29411, 0.014279], dtype=np.float32)
    #         self.cls_image_patch_info = mmcv.load(os.path.join(cls_img_patch_path, 'meta_info.pkl'))
    #         self.lesion_location = mmcv.load(lesion_location_file)
    #
    # my_class = Myclass()
    # root_path = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data/train'
    # save_dir = '/mnt/group-ai-medical-2/private/zehuigong/torch_code/ScaledYOLOv4/demo/cls_image_paste'
    # for file in os.listdir(root_path)[:10]:
    #     file_path = os.path.join(root_path, file)
    #     img, labels = read_img(file_path, my_class, need_paste=True)
    #     im_h, im_w = img.shape[:2]
    #     labels[:, 1:] = labels[:, 1:] * np.array([[im_w, im_h, im_w, im_h]], dtype=np.float32)
    #     bboxes = labels[:, 1:]
    #     xyxy = np.zeros(bboxes.shape, dtype=np.float32)
    #     xyxy[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    #     xyxy[:, 2] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    #     xyxy[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    #     xyxy[:, 3] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    #     img = mmcv.imshow_bboxes(img, xyxy, show=False)
    #     mmcv.imwrite(img, os.path.join(save_dir, file))