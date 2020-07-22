import pandas as pd
from utils.constant import TRAIN_DATA_PATH, TEST_DATA_PATH, TEST_IMAGE_PATH, SUB_PATH
import numpy as np
import torch
from data.dataset import MelanomaDataset
from torch.utils.data import DataLoader
import os
from models.effnet_model import EffNet
from models import engine
from scipy.stats import rankdata


def create_submission():
    # Create test dataset
    df = pd.read_csv(TRAIN_DATA_PATH)
    df = df[df['tfrecord'] != -1].reset_index(drop=True)  # drop duplicates

    df_test = pd.read_csv(TEST_DATA_PATH)
    df_test['target'] = np.zeros(df_test.shape[0])
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
    age_max = max(df['age_approx'].max(), df_test['age_approx'].max())
    df['age_approx'] /= age_max
    df_test['age_approx'] /= age_max
    df['age_approx'] = df['age_approx'].fillna(0)
    df_test['age_approx'] = df_test['age_approx'].fillna(0)

    df['patient_id'] = df['patient_id'].fillna(0)

    metafeatures = ['sex', 'age_approx'] + \
        [col for col in df.columns if 'site_' in col]
    metafeatures.remove('anatom_site_general_challenge')

    test_df = MelanomaDataset(TEST_IMAGE_PATH, df_test,
                              metafeatures=metafeatures, test=True)
    test_df = DataLoader(test_df, batch_size=64,
                         num_workers=6, shuffle=False)
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    print(f'Working on {device}')

    models = os.listdir('saved_models')
    results = np.zeros((df_test.shape[0], len(models)))
    for idx, model_name in enumerate(models):
        model = EffNet(nb_metafeatures=len(metafeatures))
        model.to(device)
        model.load_state_dict(torch.load(
            os.path.join('saved_models', model_name)))
        result = engine.predict(test_df, model, device)
        results[:, idx] = result
    # Ensemble
    submission = pd.read_csv(SUB_PATH)
    for j in range(len(models)):
        results[:, j] = rankdata(
            results[:, j], method='min')
    submission['target'] = results.mean(-1)
    submission.to_csv('submission.csv', index=False, float_format='%.8f')


if __name__ == '__main__':
    create_submission()
