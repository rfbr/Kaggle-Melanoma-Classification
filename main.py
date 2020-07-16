import pandas as pd
from utils.constant import TRAIN_DATA_PATH, TEST_DATA_PATH, BATCH_SIZE, LEARNING_RATE, EPOCHS, IMAGE_PATH, SIZE
from sklearn.model_selection import train_test_split
from data.dataset import MelanomaDataset
from torch.utils.data import DataLoader
import torch
from models.effnet_model import EffNet
import os
from models.engine import train, evaluation
from utils.early_stopping import EarlyStopping
from torch.cuda import amp
import numpy as np
from utils.seed import set_seed
from utils.lr import find_optimal_lr
import matplotlib.pyplot as plt
from utils.target_encoder import KFoldTargetEncoder
from utils.roc_star import epoch_update_gamma
from utils.freeze import freeze, unfreeze_layer


def main(bs=BATCH_SIZE, size=SIZE, lr=LEARNING_RATE, seed=42):
    set_seed(seed)
    df = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    # One-hot encoding of anatom_site_general_challenge feature
    concat = pd.concat([df['anatom_site_general_challenge'],
                        df_test['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True,
                             dtype=np.uint8, prefix='site')
    df = pd.concat([df, dummies.iloc[:df.shape[0]]], axis=1)
    df_test = pd.concat(
        [df_test, dummies.iloc[df.shape[0]:].reset_index(drop=True)], axis=1)

    # Sex features
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    df['sex'] = df['sex'].fillna(-1)
    df_test['sex'] = df_test['sex'].fillna(-1)

    # Age features
    df['age_approx'] /= df['age_approx'].max()
    df_test['age_approx'] /= df_test['age_approx'].max()
    df['age_approx'] = df['age_approx'].fillna(0)
    df_test['age_approx'] = df_test['age_approx'].fillna(0)

    df['patient_id'] = df['patient_id'].fillna(0)

    metafeatures = ['sex', 'age_approx'] + \
        [col for col in df.columns if 'site_' in col]
    metafeatures.remove('anatom_site_general_challenge')

    folds = sorted(df.fold.unique())
    scores = []

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    print(f'Working on {device}')

    # # Find optimal lr
    # # Define model
    # model = EffNet(nb_metafeatures=len(metafeatures))
    # model.to(device)

    # # Data loader
    # df_lr = MelanomaDataset(IMAGE_PATH.format(
    #     size=size), df, metafeatures=metafeatures)
    # df_lr = DataLoader(df_lr, batch_size=bs, num_workers=6)

    # # AMP Scaler
    # scaler = amp.GradScaler()

    # # Optimizer and scheduler
    # optimizer = torch.optim.Adam(
    #     params=model.parameters())

    # lrs, losses = find_optimal_lr(
    #     model, df_lr, optimizer, scaler, device, init_value=1e-10, final_value=1e-2)
    # plt.plot(lrs, losses)
    # plt.show()

    # CV training
    for fold in folds:
        print(f'Fold {fold}')
        # Train/valid split
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold].reset_index(drop=True)

        # Shuffle
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_valid = df_valid.sample(frac=1).reset_index(drop=True)

        # Dataloader
        df_train = MelanomaDataset(IMAGE_PATH.format(
            size=size), df_train, metafeatures=metafeatures)
        df_train = DataLoader(df_train, batch_size=bs, num_workers=6)

        # valid_images_path = [os.path.join(
        #     IMAGE_PATH.format(size=size), f'{image}.jpg') for image in df_valid['image_name']]
        df_valid = MelanomaDataset(
            IMAGE_PATH.format(size=size), df_valid, metafeatures=metafeatures, test=True)
        df_valid = DataLoader(df_valid, batch_size=bs, num_workers=6)

        # Define model
        model = EffNet(nb_metafeatures=len(metafeatures))
        model.to(device)

        # AMP Scaler
        scaler = amp.GradScaler()

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=1,
            verbose=False,
            factor=.2
        )

        # Train nd evaluation
        early_stopping = EarlyStopping(patience=3, mode='max')
        model_path = f'saved_models/model_{fold}.bin'
        best_auc = 0

        # initialize last epoch with random values
        last_whole_y_t = None
        last_whole_y_pred = None
        epoch_gamma = .2
        for epoch in range(EPOCHS):
            whole_y_t, whole_y_pred = train(epoch, epoch_gamma, last_whole_y_t, last_whole_y_pred,
                                            df_train, model, optimizer, device, scaler)
            last_whole_y_t = torch.tensor(whole_y_t).cuda()
            last_whole_y_pred = torch.tensor(whole_y_pred).cuda()
            epoch_gamma = epoch_update_gamma(
                last_whole_y_t, last_whole_y_pred, epoch)
            valid_roc_auc = evaluation(df_valid, model, optimizer, device)
            print(f'Epoch: {epoch}, validaion ROC AUC: {valid_roc_auc}')
            if valid_roc_auc > best_auc:
                best_auc = valid_roc_auc
            scheduler.step(valid_roc_auc)
            early_stopping(valid_roc_auc, model, model_path)
            if early_stopping.early_stop:
                print('Early stopping')
                break
        scores.append(best_auc)
    print(f'Cross validation score: {np.mean(scores)} +/-{np.std(scores)}')
    return np.mean(scores), np.std(scores)


if __name__ == '__main__':
    batches = [64]
    sizes = [260]  # 224, 240, 260, 380, 380, 456, 528, 600
    lrs = [5e-4]
    # f = open('cv_results.txt', 'x')
    # f.write('B1-effnet\n')
    for bs in batches:
        for size in sizes:
            # if (bs == 32 and size > 456) or (bs == 64 and size > 260) or (bs == 128 and size > 224):
            #     continue
            for lr in lrs:
                print(f"Batch size: {bs}")
                print(f"Size: {size}")
                print(f"LR: {lr}")
                # f.write(f'Batch size: {bs}, size: {size}, LR: {lr}\n')
                mean, std = main(bs=bs, size=size, lr=lr)
                # f.write(
                #     f'Cross validation score: {mean} +/-{std}\n')
    # f.close()
