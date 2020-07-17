import math
import torch
from torch.cuda import amp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


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


def find_optimal_lr(model, data_loader, optimizer, scaler, device, init_value=1e-8, final_value=10):
    nb_in_epoch = len(data_loader)-1
    update_step = (final_value/init_value)**(1/nb_in_epoch)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    best_loss = np.Inf
    losses = []
    lrs = []
    data_tqdm = tqdm(data_loader, total=len(data_loader))
    for data in data_tqdm:
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
            loss = criterion_margin_focal_binary_cross_entropy(logits, targets)
        if loss > 5 * best_loss:
            return lrs[5:-5], losses[5:-5]
        if loss < best_loss:
            best_loss = loss

        losses.append(loss)
        lrs.append(lr)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr *= update_step
        optimizer.param_groups[0]['lr'] = lr

    return lrs[10:-5], losses[10:-5]
