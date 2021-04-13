import pandas as pd
import numpy as np
import cv2
import torch
import timm
from multilabel_dataset import TestDataset
from multilabel_util import *
from albumentations import Compose, Normalize

csv_path = '../input/vinbigdata-chest-xray-abnormalities-detection/test_meta.csv'
image_path = '../input/test1024/' # The path to the folder with converted PNG files
model_path = '../classifier_weights/1024/'
save_path = '../classifier_preds/1024/'

bs = 2
IMG_SIZE = 1024

def main():
    test = pd.read_csv(csv_path)

    test_transform = Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
    ])

    for fold in range(5):

        epoch_cls_dict = {}

        for cls in range(15):
            epoch = torch.load(
                model_path + f'multilabel_efnb4_v1_cls{cls}_fold{fold}.pth'
            )['epoch']
            try:
                epoch_cls_dict[epoch].append(cls)
            except:
                epoch_cls_dict[epoch] = [cls]

        valset = TestDataset(
            test,
            image_path = image_path,
            transform=test_transform
        )
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=bs, shuffle=False, num_workers=1
        )

        model = timm.create_model('tf_efficientnet_b4_ns',pretrained=True,num_classes=15).cuda()    

        preds = np.zeros((3000,15))

        for k,v in epoch_cls_dict.items(): 
            model.load_state_dict(
                torch.load(
                    model_path + f'multilabel_efnb4_v1_cls{v[0]}_fold{fold}.pth'
                )['weight']
            )
            p = test_model(model, val_loader)
            for i in v:
                preds[:,i] = p[:,i]    

        df_preds = pd.DataFrame(
            preds, columns=np.arange(0,15)
        ).assign(image_id = test['image_id'])

        df_preds.to_csv(
            save_path+f'multilabel_efnb4_fold{fold}_preds.csv',
            index=False
        )
                    
if __name__=='__main__':
    main()    