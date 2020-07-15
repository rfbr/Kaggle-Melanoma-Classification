import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold

if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    df['fold'] = -1
    X = df.drop('target', axis=1)
    y = df['target']

    kfold = GroupKFold(n_splits=5)
    for fold, (_, val_idx) in enumerate(kfold.split(X=X, y=y, groups=X['patient_id'].tolist())):
        df.loc[val_idx, 'fold'] = fold
    df.to_csv('../data/train_folds.csv', index=False)
