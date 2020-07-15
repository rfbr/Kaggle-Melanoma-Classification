import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed
import sys
ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize(image_path, output_folder, size):
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize(size, resample=Image.LANCZOS)
    img.save(output_path)


if __name__ == '__main__':
    for n in [240, 260, 300, 380, 456, 528, 600]:
        SIZE = (n, n)
        input_folder = '../data/jpeg/train'
        if not os.path.isdir(f'../data/train_{n}'):
            os.mkdir(f'../data/train_{n}')
        output_folder = f'../data/train_{n}'
        images = glob.glob(os.path.join(input_folder, "*.jpg"))
        Parallel(n_jobs=12)(
            delayed(resize)(
                i,
                output_folder,
                SIZE
            ) for i in tqdm(images)
        )
