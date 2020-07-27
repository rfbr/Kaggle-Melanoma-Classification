import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.cuda import amp
from tqdm import tqdm

from utils.averagemeter import AverageMeter
from utils.losses import margin_focal_binary_cross_entropy


def train(data_loader, model, optimizer, device, scaler):
    model.train()
    losses = AverageMeter()
    tqdm_data = tqdm(data_loader, total=len(data_loader))
    for data in tqdm_data:
        # Fetch data
        images = data['image']
        metadata = data['metadata']
        targets = data['target']
        images = images.to(device, dtype=torch.float32)
        metadata = metadata.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)

        # Compute model output
        optimizer.zero_grad()
        with amp.autocast():
            logits = model(images, metadata)
            loss = margin_focal_binary_cross_entropy(
                logits, targets)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item(), images.size(0))

        tqdm_data.set_postfix(loss=losses.avg)
    torch.cuda.empty_cache()


def evaluation(data_loader, model, device):
    model.eval()
    prediction_list = []
    target_list = []
    with torch.no_grad():
        for data in data_loader:
            # Fetch data
            images = data['image']
            metadata = data['metadata']
            targets = data['target']
            for i in range(len(images)):
                images[i] = images[i].to(device, dtype=torch.float32)
            metadata = metadata.to(device, dtype=torch.float32)

            # Compute model output
            tta_preds = []
            for image in images:
                logits = model(image, metadata)
                preds = torch.sigmoid(logits).cpu(
                ).detach().numpy().reshape(-1, 1)
                tta_preds.append(preds)
            prediction_list.append(
                np.median(np.concatenate(tta_preds, -1), -1))
            target_list.append(targets.cpu().detach().numpy())
    torch.cuda.empty_cache()

    roc_auc = roc_auc_score(np.concatenate(target_list),
                            np.concatenate(prediction_list))
    return roc_auc


def predict(data_loader, model, device):
    model.eval()

    prediction_list = []
    tqdm_data = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for data in tqdm_data:
            # Fetch data
            images = data['image']
            metadata = data['metadata']
            for i in range(len(images)):
                images[i] = images[i].to(device, dtype=torch.float32)
            metadata = metadata.to(device, dtype=torch.float32)

            # Compute model output
            tta_preds = []
            for image in images:
                logits = model(image, metadata)
                preds = torch.sigmoid(logits).cpu(
                ).detach().numpy().reshape(-1, 1)
                tta_preds.append(preds)
            prediction_list.append(np.concatenate(tta_preds, -1).mean(-1))
    torch.cuda.empty_cache()
    return np.concatenate(prediction_list)
