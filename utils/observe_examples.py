import pandas as pd
import os
from constant import DATA_PATH
import matplotlib.pyplot as plt
from PIL import Image
IMAGE_PATH = '/home/romain/Projects/Kaggle/melanoma_classification/data/jpeg/train'
if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    df_pos = df[df['target'] == 1].reset_index(drop=True)
    df_neg = df[df['target'] == 0].reset_index(drop=True)
    for row in df_neg.sample(n=10).iterrows():
        img = Image.open(os.path.join(
            IMAGE_PATH, row[1][0]+'.jpg'))
        plt.imshow(img)
        plt.show()
