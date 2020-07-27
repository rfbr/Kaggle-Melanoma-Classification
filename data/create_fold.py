import os

import pandas as pd
from sklearn.model_selection import GroupKFold


def create_k_fold(path: str, n_split: int):
    """
    Create k fold training csv, grouped by patient.
    """
    df = pd.read_csv(path)
    df['fold'] = -1
    X = df.drop('target', axis=1)
    y = df['target']

    kfold = GroupKFold(n_splits=n_split)
    for fold, (_, val_idx) in enumerate(kfold.split(X=X, y=y, groups=X['patient_id'].tolist())):
        df.loc[val_idx, 'fold'] = fold
    df.to_csv(os.path.join(os.path.dirname(path),
                           'train_fold.csv'), index=False)


if __name__ == '__main__':
    create_k_fold(path='../external_data/train_concat.csv', n_split=5)
