from utils.averagemeter import AverageMeter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
from torch.cuda import amp


def criterion_margin_focal_binary_cross_entropy(logit, truth):
    weight_pos = 2
    weight_neg = 1
    gamma = 2
    margin = 0.2
    em = np.exp(margin)

    logit = logit.view(-1)
    truth = truth.view(-1)
    log_pos = -F.logsigmoid(logit)
    log_neg = -F.logsigmoid(-logit)

    log_prob = truth*log_pos + (1-truth)*log_neg
    prob = torch.exp(-log_prob)
    margin = torch.log(em + (1-em)*prob)

    weight = truth*weight_pos + (1-truth)*weight_neg
    loss = margin + weight*(1 - prob) ** gamma * log_prob

    loss = loss.mean()
    return loss


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
            # loss = nn.BCEWithLogitsLoss()(logits, targets)
            loss = criterion_margin_focal_binary_cross_entropy(logits, targets)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item(), images.size(0))
        tqdm_data.set_postfix(loss=losses.avg)
    torch.cuda.empty_cache()


def evaluation(data_loader, model, optimizer, device):
    model.eval()
    prediction_list = []
    target_list = []
    with torch.no_grad():
        for data in data_loader:
            # Fetch data
            images = data['image']
            metadata = data['metadata']
            targets = data['target']
            images = images.to(device, dtype=torch.float32)
            metadata = metadata.to(device, dtype=torch.float32)

            # Compute model output
            logits = model(images, metadata)
            preds = torch.sigmoid(logits).cpu().detach().numpy().reshape(-1, 1)
            prediction_list.append(preds)
            target_list.append(targets.cpu().detach().numpy())
    torch.cuda.empty_cache()
    roc_auc = roc_auc_score(np.concatenate(target_list),
                            np.concatenate(prediction_list))
    return roc_auc
