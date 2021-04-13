import pandas as pd
import numpy as np
import cv2
import torch
import timm
from copy import deepcopy
from multilabel_dataset import TrainDataset
from multilabel_util import *
from albumentations import (
    Compose,
    Normalize,
    ShiftScaleRotate,
    RandomBrightnessContrast,
    MotionBlur,
    CLAHE,
    HorizontalFlip
)

csv_path = '../input/vinbigdata-chest-xray-abnormalities-detection/multilabel_cls_train.csv'
pos_weight_path = '../input/vinbigdata-chest-xray-abnormalities-detection/multilabel_pos_weight.npy'
image_path = '../input/train1024/' # The path to the folder with converted PNG files
save_path = '../classifier_weights/1024/'

bs = 2
lr = 1e-3
N_EPOCHS = 100
IMG_SIZE = 1024
    
def main():
    seed_everything(42)

    train = pd.read_csv(csv_path)
    cls_pos_weights = np.load(pos_weight_path)

    train_transform = Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(scale_limit = 0.15, rotate_limit = 10, p = 0.5),
        RandomBrightnessContrast(p=0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
    ])
    test_transform = Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
    ])

    for fold in range(5):
        trainset = TrainDataset(
            train.loc[train['fold']!=fold].reset_index(),
            image_path = image_path,
            transform=train_transform
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=bs, num_workers=1,
            shuffle=True 
        )

        valset = TrainDataset(
            train.loc[train['fold']==fold].reset_index(),
            image_path = image_path,
            transform=test_transform
        )
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=bs, shuffle=False, num_workers=1
        )

        model = timm.create_model('tf_efficientnet_b4_ns',pretrained=True,num_classes=15).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight = torch.FloatTensor(cls_pos_weights).cuda() 
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.1, mode='max')

        best_weights = deepcopy(model.state_dict())
        previous_lr = lr
        best_auc = 0
        best_aucs = [0]*15
        best_val_loss = 100
        es = 0

        for epoch in range(N_EPOCHS):
            avg_loss = train_model(model, train_loader, optimizer, criterion)
            avg_val_loss, aucs, auc = val_model(model, val_loader, optimizer, criterion, cls_pos_weights)

            print(
                'epoch:', epoch, 'lr:', previous_lr, 'val_loss:',avg_val_loss, 'weighted avg auc:',auc
            )
            print('aucs:',aucs)

            # Record the best weights if either of AUC or val_loss improved.
            if auc > best_auc or avg_val_loss < best_val_loss:
                print('saving best weight...')
                best_weights = deepcopy(model.state_dict())
                for k,v in best_weights.items():
                    best_weights[k] = v.cpu()

            # Save the model weight if the AUC of any class is improved. 
            for i in range(len(best_aucs)):
                if aucs[i] > best_aucs[i]:
                    best_aucs[i] = aucs[i]
                    d = {
                        'weight':model.state_dict(),
                        'auc':aucs[i],
                        'epoch':epoch,
                    }
                    torch.save(
                        d, save_path + f'multilabel_efnb4_v1_cls{i}_fold{fold}.pth'
                    )

            # Update best avg_val_loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            # Update best weighted../../results/multilabel_cls/v2/ AUC and implement early stop
            if auc > best_auc:
                es = 0
                best_auc = auc
            else:
                es += 1
                if es > 10:
                    break

            scheduler.step(auc)  

            # if lr changes, start from previous best weight:
            if optimizer.param_groups[0]['lr'] < previous_lr:
                print('restoring best weight...')
                model.load_state_dict(best_weights)
                previous_lr = optimizer.param_groups[0]['lr']

                if optimizer.param_groups[0]['lr'] < 0.99e-6:
                    break
                    
if __name__=='__main__':
    main()    