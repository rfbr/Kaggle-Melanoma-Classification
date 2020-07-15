import cv2
from PIL import Image
import numpy as np
IMAGE_PATH = '/home/romain/Projects/Kaggle/melanoma_classification/data/train_224/ISIC_0015719.jpg'


class MicroscopeAugmentation:
    def __init__(self, p=.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            x1, y1, x2, y2 = img.size[0]//4, img.size[1]//4, (3 *
                                                              img.size[0])//4, (3*img.size[1])//4
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_img = cropped_img.resize((224, 224), Image.LANCZOS)
            np_cropped_img = np.array(cropped_img)
            circle = cv2.circle((np.ones(np_cropped_img.shape) * 255).astype(np.uint8),
                                (np_cropped_img.shape[0]//2,
                                 np_cropped_img.shape[1]//2),
                                np.random.randint(
                                    np_cropped_img.shape[0]//2 - 3, np_cropped_img.shape[0]//2 + 15),
                                (0, 0, 0),
                                -1)

            mask = circle - 255
            np_cropped_img = np.multiply(np_cropped_img, mask)
            aug_img = Image.fromarray(np_cropped_img)
            return aug_img
        else:
            return img


if __name__ == '__main__':
    img = Image.open(IMAGE_PATH)
    # img.show()
    # x1, y1, x2, y2 = img.size[0]//4, img.size[1]//4, (3 *
    #                                                   img.size[0])//4, (3*img.size[1])//4
    # cropped_img = img.crop((x1, y1, x2, y2))
    # cropped_img = cropped_img.resize((224, 224), Image.LANCZOS)
    # np_cropped_img = np.array(cropped_img)
    # circle = cv2.circle((np.ones(np_cropped_img.shape) * 255).astype(np.uint8),
    #                     (np_cropped_img.shape[0]//2,
    #                      np_cropped_img.shape[1]//2),
    #                     np.random.randint(
    #                         np_cropped_img.shape[0]//2 - 3, np_cropped_img.shape[0]//2 + 15),
    #                     (0, 0, 0),
    #                     -1)

    # mask = circle - 255
    # np_cropped_img = np.multiply(np_cropped_img, mask)
    # cropped_img = Image.fromarray(np_cropped_img)
    # cropped_img.show()
    mic_aug = MicroscopeAugmentation()
    mic_aug(img).show()
