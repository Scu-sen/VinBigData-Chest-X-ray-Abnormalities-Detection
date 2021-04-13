import pandas as pd
import numpy as np
import cv2
import torch
import random
import os
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    
def train_model(model, train_loader, optimizer, criterion): # train 1 epoch
    """
    Trains the model for 1 epoch
    
    Parameters:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): Dataloader object for training.
        optimizer (A torch.optim class): The optimizer.
        criterion (A function in torch.nn.modules.loss): The loss function. 
        
    Return: 
        avg_loss (float): The average loss.
    """
    
    model.train() 
    
    losses = AverageMeter()
    avg_loss = 0.

    optimizer.zero_grad()
    
    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(tk):
        imgs_train, labels_train = imgs.cuda(), labels.cuda()
        output_train = model(imgs_train)
        
        loss = criterion(output_train, labels_train) 
        loss.backward()

        optimizer.step() 
        optimizer.zero_grad() 
        
        avg_loss += loss.item() / len(train_loader)
        losses.update(loss.item(), imgs_train.size(0))
        tk.set_postfix(loss=losses.avg)
        
    return avg_loss


def val_model(model, val_loader, optimizer, criterion, cls_pos_weights):
    """
    Test the model on the validation set
    
    Parameters:
        model (torch.nn.Module): The model to be trained.
        val_loader (torch.utils.data.DataLoader): Dataloader object for validation.
        optimizer (A torch.optim class): The optimizer.
        criterion (A torch.nn.modules.loss class): The loss function. 
        cls_pos_weights (np.array): An array with shape (15,) represented the pos_weight of each class.
        
    Return: 
        avg_val_loss (float): The average loss.
        aucs (np.array): The validation AUC of each class.
        weighted_auc (float): Weighted average of AUCs (based on cls_pos_weights)
    """
    model.eval()
    
    losses = AverageMeter()
    avg_val_loss = 0.
    valid_preds, valid_targets = [], []
    top1 = []
    
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        
        for idx, (imgs, labels) in enumerate(tk):
            
            imgs_valid, labels_valid = imgs.cuda(), labels.cuda()
            output_valid = model(imgs_valid)
            
            loss = criterion(output_valid, labels_valid)
            avg_val_loss += loss.item() / len(val_loader)
            losses.update(loss.item(), imgs_valid.size(0))
            tk.set_postfix(loss=losses.avg)
            
            valid_pred = torch.sigmoid(output_valid).detach().cpu().numpy()
            label_valid = labels_valid.detach().cpu().numpy()
         
            valid_preds.append(valid_pred)
            valid_targets.append(label_valid.round().astype(int))

        valid_preds = np.concatenate(valid_preds,axis=0).T
        valid_targets = np.concatenate(valid_targets,axis=0).T
        
        aucs = np.array(
            [roc_auc_score(i,j) if len(set(i))>1 else np.nan for i,j in zip(valid_targets, valid_preds)]
        )
        
        weighted_auc = np.nansum(cls_pos_weights * aucs)/np.nansum(cls_pos_weights)
    return avg_val_loss, aucs, weighted_auc


def test_model(model, val_loader):    
    """
    Inference using the model on the test set
    
    Parameters:
        model (torch.nn.Module): The model to be trained.
        val_loader (torch.utils.data.DataLoader): Dataloader object for validation.

    Return: 
        preds (np.array): An array with shape (num_samples, 15).
    """
    model.eval()
    
    preds = []
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (imgs) in enumerate(tk):
            imgs_valid = imgs.cuda()
            output_valid = model(imgs_valid)
            
            valid_pred = torch.sigmoid(output_valid).detach().cpu().numpy()
         
            preds.append(valid_pred)

        preds = np.concatenate(preds,axis=0)
            
    return preds