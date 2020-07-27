import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from constant import TRAIN_DATA_PATH, TRAIN_IMAGE_PATH

if __name__ == '__main__':
    df = pd.read_csv(TRAIN_DATA_PATH)
    df_pos = df[df['target'] == 1].reset_index(drop=True)
    df_neg = df[df['target'] == 0].reset_index(drop=True)
    for row in df_neg.sample(n=10).iterrows():
        img = Image.open(os.path.join(
            TRAIN_IMAGE_PATH, row[1][0]+'.jpg'))
        plt.imshow(img)
        plt.show()
