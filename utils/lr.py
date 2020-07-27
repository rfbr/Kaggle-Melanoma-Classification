import numpy as np
import torch
from torch.cuda import amp
from tqdm import tqdm


def find_optimal_lr(model, data_loader, optimizer, scaler, device, init_value=1e-8, final_value=10):
    """
    Estimate an optimal learning rate.
    For one epoch, try different LR for each batch size from init value to final value,
    thus we can choose the LR corresponding to the steepest point in the curve.
    """
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
