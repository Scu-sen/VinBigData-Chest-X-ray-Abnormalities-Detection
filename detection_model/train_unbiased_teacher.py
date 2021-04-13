import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict

import test  # import test.py to get mAP after each epoch
from train import save_model
from models.yolo import Model
from utils.datasets import create_dataloader, GroupedSemiSupDatasetTwoCrop
import utils.comm as comm
from utils.general import (
    check_img_size, torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors,
    labels_to_image_weights, compute_loss, plot_images, fitness, strip_optimizer, plot_results,
    get_latest_run, check_git_status, check_file, increment_dir, print_mutation, plot_evolution,
    not_increment_dir, non_max_suppression)
from utils.google_utils import attempt_download
from utils.torch_utils import init_seeds, ModelEMA, select_device, intersect_dicts
from utils.aafmadet.show_coco_annotations import draw_bbox
import mmcv


class Process_prediction(object):
    def __init__(self, stride, nl, na, anchor_grid, nc=80):
        super(Process_prediction, self).__init__()
        self.stride = stride
        self.nl = nl  # number of detection layers
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = anchor_grid

    def __call__(self, x):
        z = []
        for i in range(self.nl):
            bs, na, ny, nx, no = x[i].shape

            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, no))

        return torch.cat(z, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def build_model(opt, weights, rank, nc):
    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=opt.inp_channel, nc=nc).to(device)  # create
        exclude = ['anchor'] if opt.cfg else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report

        return model, (ckpt, state_dict)
    else:
        model = Model(opt.cfg, ch=opt.inp_channel, nc=nc).to(device)  # create
        # model = model.to(memory_format=torch.channels_last)  # create
        return model, None


def update_teacher_model(model, model_teacher, keep_rate):
    if comm.get_world_size() > 1:
        student_model_dict = {
            key[7:]: value for key, value in model.state_dict().items()
        }
    else:
        student_model_dict = model.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    model_teacher.load_state_dict(new_teacher_dict)
    return True


def process_pesudo_label(prediction, img_shape, conf_thres):
    """
    process the network output, to get target format
    Args:
        prediction (list[Tensor]): (k, 6) [x1, y1, x2, y2, score, cls]
           !!!Note that element of prediction maybe None, indicating
              that this image has no prediction!!!!!!
        img_shape (tuple): (h, w)
        conf_thres (float): prediction confidence threshold e.g. 0.7
    Return:
        Tensor: (n, 6) [img_i, cls_id, ct_x, ct_y, bw, bh](normalized)
    """
    labels_out = []
    for i, pred in enumerate(prediction):
        if pred is None:
            labels = torch.zeros((0, 6))
            labels_out.append(labels)
            continue
        inds = pred[:, 4] > conf_thres
        left_pred = pred[inds, :]
        nL = left_pred.shape[0]

        labels = torch.zeros((nL, 6))
        labels[:, 0] = i
        labels[:, 1] = left_pred[:, -1]
        labels[:, 2:4] = (left_pred[:, 0:2] + left_pred[:, 2:4]) * 0.5
        labels[:, 4:6] = left_pred[:, 2:4] - left_pred[:, 0:2]
        labels[:, 2:6:2] /= img_shape[1]  # normalize ct_x, bw
        labels[:, 3:6:2] /= img_shape[0]  # normalize ct_y, bh

        labels_out.append(labels)
    labels_out = torch.cat(labels_out, 0)
    return labels_out


def visualize_pseudo_label(labels, imgs, paths):
    """check the correctness of the online generated pseudo label"""
    save_dir = '/mnt/group-ai-medical-2/private/zehuigong/dataset1/A_AFMA_Detection/demo'
    labels = labels.data.cpu().numpy()
    imgs = imgs.data.cpu().numpy()
    c, h, w = imgs.shape[1:]
    imgs = imgs[:, 1:4, :, :]
    num_imWith_label = 0
    for i, im in enumerate(imgs):
        idxes = np.where(labels[:, 0] == i)[0]
        if idxes.shape[0] < 1:
            continue
        num_imWith_label += 1
        img_labels = labels[idxes, 1:].astype(np.float32)
        img_labels[:, 1::2] *= w
        img_labels[:, 2::2] *= h
        new_labels = np.zeros((img_labels.shape[0], 4), dtype=np.float32)
        new_labels[:, :2] = img_labels[:, 1:3] - img_labels[:, 3:5] * 0.5
        new_labels[:, 2:] = img_labels[:, 1:3] + img_labels[:, 3:5] * 0.5
        im = (im * 255).transpose((1, 2, 0))[:, :, ::-1]  # h, w, c --> RGB to BGR
        filename = '_'.join(paths[i].split('/')[-2:])
        im = draw_bbox(im.astype(np.uint8), new_labels.astype(np.int), img_labels[:, 0])
        mmcv.imwrite(im, os.path.join(save_dir, 'pseudo_label_' + filename))
    print('there are total {} imgs in one batch, {} with label'.format(imgs.shape[0],  num_imWith_label))


def train(hyp, opt, device, tb_writer=None):
    print(f'Hyperparameters {hyp}')
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve'  # logging directory
    wdir = str(log_dir / 'weights') + os.sep  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    results_file = str(log_dir / 'results.txt')
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # TODO: Use DDP logging. Only the first process is allowed to log.
    # Save run settings
    with open(log_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(log_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_label_ann_file = data_dict['train_label_ann'].format(opt.rad_id, opt.fold)
    train_unlabel_ann_file = data_dict['train_unlabel_ann'].format(opt.rad_id, opt.fold)
    test_ann_file = data_dict['val_ann'].format(opt.rad_id, opt.fold)
    train_root = data_dict['train_img_root']
    test_root = data_dict['val_img_root']
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # create student model
    model, ckpt_info = build_model(opt, weights, rank, nc)
    # create teacher model
    model_teacher, _ = build_model(opt, weights, rank, nc)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if ckpt_info is not None:
        ckpt, state_dict = ckpt_info
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
        # start_epoch = ckpt['epoch'] + 1  ################################################
        # if epochs < start_epoch:
        print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
              (weights, ckpt['epoch'], epochs))
        epochs += start_epoch  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        model_teacher = torch.nn.DataParallel(model_teacher)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print('Using SyncBatchNorm()')

    # Exponential moving average
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=(opt.local_rank))

    # Trainloader
    dataloader = GroupedSemiSupDatasetTwoCrop(
        train_label_ann_file, train_unlabel_ann_file,
        img_root=train_root,
        img_size=imgsz,
        batch_size=batch_size,
        stride=gs,
        opt=opt,
        hyp=hyp,
        augment=True,
        cache=opt.cache_images, rect=opt.rect, local_rank=rank,
        world_size=opt.world_size,
        image_weights=data_dict.get('image_weights', False))
    mlc = np.concatenate(dataloader.label_dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Testloader
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates ***
        # local_rank is set to -1. Because only the first process is expected to do evaluation.
        testloader = \
        create_dataloader(test_ann_file, test_root, imgsz_test, 16, gs, opt, hyp=hyp, augment=False,
                          cache=opt.cache_images, rect=True, local_rank=-1, world_size=opt.world_size,
                          keep_no_anno_img=opt.keep_no_anno_img, split='val')[0]

    # Model parameters
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataloader.label_dataset.labels, nc).to(device)  # attach class weights
    model.names = names

    # Class frequency
    if rank in [-1, 0]:
        labels = np.concatenate(dataloader.label_dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.
        # model._initialize_biases(cf.to(device))
        plot_labels(labels, save_dir=log_dir)
        if tb_writer:
            tb_writer.add_histogram('classes', c, 0)

        # Check anchors
        if not opt.noautoanchor:
            check_anchors(dataloader.label_dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Get the predictions from network output
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    get_pred = Process_prediction(model.module.stride if hasattr(model, 'module') else model.stride,
                                  nl=m.nl, na=m.na, anchor_grid=m.anchor_grid, nc=model.nc)

    # Start training
    t0 = time.time()
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    if rank in [0, -1]:
        print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
        print('Using %g dataloader workers' % dataloader.num_workers)
        print('Starting training for %g epochs...' % epochs)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.label_dataloader.sampler.set_epoch(epoch)
            dataloader.unlabel_dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(zip(dataloader.label_dataloader, dataloader.unlabel_dataloader))
        if rank in [-1, 0]:
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (label_data, unlabel_data) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)

            l_imgs, l_ims_strong, l_targets, l_path, _ = label_data
            ul_imgs, ul_ims_strong, ul_targets, ul_path, _ = unlabel_data

            # update teacher model using EMA of student weight
            if ni > 0 and ni % opt.teacher_up_iter == 0:
                update_teacher_model(model, model_teacher, keep_rate=opt.pse_ema_rate)

            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            if rank in [-1, 0]:  # only test on gpu 0
                with torch.no_grad():
                    ul_imgs = ul_imgs.to(device, non_blocking=True).float()
                    inf_out = model_teacher(ul_imgs)
                    inf_out = get_pred(inf_out)
                    # list[Tensor](k, 6): (x1, y1, x2, y2, score, cls)
                    output = non_max_suppression(
                        inf_out, conf_thres=0.001, iou_thres=0.5, merge=False)
                    # process the format of prediction to targets
                    ul_targets = process_pesudo_label(output,
                                                      img_shape=ul_imgs.shape[2:],
                                                      conf_thres=opt.pse_conf_thres)
                    # print('visualize the pseudo label')
                    # visualize_pseudo_label(ul_targets, ul_imgs, ul_path)

            # input both strong and weak augmentation to model
            # imgs = torch.cat([l_imgs, l_ims_strong], dim=0)
            # l_targets_strong = l_targets.clone()
            # l_targets_strong[:, 0] += l_imgs.shape[0]
            # targets = torch.cat([l_targets, l_targets_strong], dim=0)

            # input only weak augmentation image to model
            imgs = l_imgs
            targets = l_targets

            imgs = imgs.to(device, non_blocking=True).float()
            ul_imgs_strong = ul_imgs.to(device, non_blocking=True).float()

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.7, imgsz * 1.1 + gs) // gs * gs  # size (0.5, 1.5)
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Autocast
            with amp.autocast(enabled=cuda):
                # Forward all_label_data and compute Loss
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size

                # forward pseudo label data and compute loss
                ul_pred = model(ul_imgs_strong)
                lobj, lcls, _ = compute_loss(ul_pred, ul_targets.to(device), model, return_each=True)

                loss = loss + (lobj + lcls) * opt.pse_unsup_lw

                loss_items[1] += lobj.detach()[0] * opt.pse_unsup_lw
                loss_items[2] += lcls.detach()[0] * opt.pse_unsup_lw
                loss_items[3] += (lobj.detach()[0] + lcls.detach()[0]) * opt.pse_unsup_lw
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                # if not torch.isfinite(loss):
                #     print('WARNING: non-finite loss, ending training ', loss_items)
                #     return results

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if ni < 3:
                    f = str(log_dir / ('train_batch%g.jpg' % ni))  # filename
                    result = plot_images(images=imgs, targets=targets, paths=l_path + l_path, fname=f)
                    if tb_writer and result is not None:
                        tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema is not None:
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            final_epoch = epoch + 1 == epochs
            if final_epoch or (epoch % opt.test_freq == 0):  # Calculate mAP  not opt.notest or
                results, maps, times = test.test(opt.data,
                                                 batch_size=batch_size,
                                                 imgsz=imgsz_test,
                                                 save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                                 model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=log_dir,
                                                 high_thr=opt.high_thr if opt.keep_no_anno_img else None)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Tensorboard
            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
            if fi > best_fitness:
                best_fitness = fi
                save_model(ema, results_file, epoch, fi, optimizer, opt, best)

            # Save model
            if (epoch > 0 and epoch % opt.save_freq == 0) or (final_epoch and not opt.evolve):
                save_model(ema, results_file, epoch, best_fitness, optimizer, opt,
                           last.replace('.pt', '_{:03d}.pt'.format(epoch)))
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        n = ('_' if len(opt.name) and not opt.name.isnumeric() else '') + opt.name
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2, f2.replace('.pt', '_strip.pt')) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload
        # Finish
        if not opt.evolve:
            plot_results(save_dir=log_dir)  # save as results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov4-p5.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const='get_last', default=False,
                        help='resume from given path/last.pt, or most recent run if blank')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--save_freq', type=int, default=1, help='save model frequency.')
    parser.add_argument('--save_optim', action='store_true', help='store the optimizer in checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--test_freq', type=int, default=1, help='testing frequency.')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi_scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--rad_id', type=str, default='rad_id9', help="which rad's annotations we want to train on")
    parser.add_argument('--fold', type=int, default=1, help="which fold of K fold cross-validation.")
    parser.add_argument('--keep_no_anno_img', action='store_true',
                        help='whether using images with no annotations to train')
    parser.add_argument('--high_thr', type=float, default=None, help='high threshold of two-threshold method')
    parser.add_argument('--resam_normal_freq', type=int, default=-1,
                        help='how many epoch to resample the normal image ids')
    parser.add_argument('--mixup_ratio', type=float, default=0.2, help='mixup augmentation ratio')
    parser.add_argument('--inp_channel', type=int, default=3, help='number of input channel, default=3')

    # parameters for unbiased-teacher
    parser.add_argument('--pse_conf_thres', type=float, default=0.7, help="confidence threshold for pseudo label")
    parser.add_argument('--pse_unsup_lw', type=float, default=4, help="unsupervised loss weight")
    parser.add_argument('--pse_ema_rate', type=float, default=0.9996, help="ema rate for updating params of teacher model")
    parser.add_argument('--teacher_up_iter', type=int, default=1, help='how many iters to update teacher model')

    opt = parser.parse_args()

    # Resume
    if opt.resume:
        last = get_latest_run() if opt.resume == 'get_last' else opt.resume  # resume from most recent run
        if last and not opt.weights:
            print(f'Resuming training from {last}')
        opt.weights = last if opt.resume and not opt.weights else opt.weights
    if opt.local_rank == -1 or ("RANK" in os.environ and os.environ["RANK"] == "0"):
        check_git_status()

    opt.hyp = opt.hyp or ('data/hyp.finetune.yaml' if opt.weights else 'data/hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'

    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    device = select_device(opt.device, batch_size=opt.batch_size)
    opt.total_batch_size = opt.batch_size
    opt.world_size = 1
    opt.global_rank = -1

    # DDP mode
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        opt.world_size = dist.get_world_size()
        opt.global_rank = dist.get_rank()
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    print(opt)
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
    #################change the hyp-parameter#############
    # hyp['fl_gamma'] = opt.mixup_ratio

    # Train
    if not opt.evolve:
        tb_writer = None
        if opt.global_rank in [-1, 0]:
            print('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % opt.logdir)
            # tb_writer = SummaryWriter(log_dir=increment_dir(Path(opt.logdir) / 'exp', opt.name))  # runs/exp
            tb_writer = SummaryWriter(log_dir=not_increment_dir(Path(opt.logdir) / 'exp', opt.name))  # runs/exp

        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'momentum': (0.1, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'giou': (1, 0.02, 0.2),  # GIoU loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (1, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (0, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (1, 0.0, 1.0),  # image flip left-right (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path('runs/evolve/hyp_evolved.yaml')  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(100):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.9, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print('Hyperparameter evolution complete. Best results saved as: %s\nCommand to train a new model with these '
              'hyperparameters: $ python train.py --hyp %s' % (yaml_file, yaml_file))
