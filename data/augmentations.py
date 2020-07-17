import time
from tqdm import tqdm
from torchvision import transforms
import cv2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms as plt_transforms
from PIL import Image
import io
import warnings
from copy import deepcopy
import imutils
warnings.filterwarnings("ignore")


def rotPt(x, y, angle):
    cx = 260//2
    cy = 260//2

    x -= cx
    y -= cy

    tmp = x
    x = x*np.cos(angle) - y*np.sin(angle)
    y = tmp*np.sin(angle) + y*np.cos(angle)

    x += cx
    y += cy

    return x, y


class MicroscopeAugmentation:
    def __init__(self, p=.5, size=(224, 224)):
        self.p = p
        self.size = size

    def __call__(self, img):
        if np.random.rand() < self.p:
            x1, y1, x2, y2 = img.size[0]//4, img.size[1]//4, (3 *
                                                              img.size[0])//4, (3*img.size[1])//4
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_img = cropped_img.resize(self.size, Image.LANCZOS)
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


class HairAugmentation:
    def __init__(self, p=.5, n_hair_range=(20, 60), degrees=[2, 3, 4, 5, 6]):
        self.p = p
        self.degrees = degrees
        self.n_hair_range = n_hair_range

    def __call__(self, img):
        if np.random.rand() < self.p:
            n_hair = np.random.randint(*self.n_hair_range)
            img_shape = img._size
            plt.imshow(img)
            for _ in range(n_hair):

                # Random rectangle
                x_rectangle = np.random.randint(
                    low=0, high=img_shape[0]-1, size=2)
                y_rectangle = np.random.randint(
                    low=0, high=img_shape[1]-1, size=2)

                x_min = min(x_rectangle[0], x_rectangle[1])
                x_max = max(x_min+1, max(x_rectangle[0], x_rectangle[1]))
                y_min = min(y_rectangle[0], y_rectangle[1])
                y_max = max(y_min+1, max(y_rectangle[0], y_rectangle[1]))

                X = np.random.randint(low=x_min, high=x_max,
                                      size=50).reshape((-1, 1))
                y = np.random.randint(low=y_min, high=y_max, size=50)

                X_plot = np.linspace(x_min, x_max, 100).reshape((-1, 1))
                degree = np.random.choice(self.degrees)
                model = make_pipeline(PolynomialFeatures(degree), Ridge())
                model.fit(X, y)
                hair = model.predict(X_plot)
                # first of all, the base transformation of the data points is needed
                base = plt.gca().transData
                random_degree = np.random.randint(0, 360)
                rot = plt_transforms.Affine2D().rotate_deg_around(
                    img_shape[0]//2, img_shape[1]//2, random_degree)
                alpha = np.random.rand()*.4
                plt.plot(X_plot, hair, 'black',
                         alpha=alpha, transform=base+rot)

            buf = io.BytesIO()
            plt.savefig(buf, format='jpg')
            aug_image = Image.open(buf).copy()
            buf.close()
            aug_image = aug_image.resize((260, 260))
            return aug_image
        else:
            return img


class CVHairAugmentation:
    def __init__(self, p=.5, n_hair_range=(20, 70), degrees=[2, 3, 4, 5, 6]):
        self.p = p
        self.degrees = degrees
        self.n_hair_range = n_hair_range

    def __call__(self, img):
        if np.random.rand() < self.p:
            # PIL image to openCV image
            img = img.convert('RGBA')
            open_cv_image = np.array(img)
            background = cv2.cvtColor(
                open_cv_image, cv2.COLOR_RGB2BGRA)

            n_hair = np.random.randint(*self.n_hair_range)
            img_shape = img._size

            for _ in range(n_hair):
                # Random rectangle
                x_rectangle = np.random.randint(
                    low=0, high=img_shape[0]-1, size=2)
                y_rectangle = np.random.randint(
                    low=0, high=img_shape[1]-1, size=2)

                x_min = min(x_rectangle[0], x_rectangle[1])
                x_max = max(x_min+1, max(x_rectangle[0], x_rectangle[1]))
                y_min = min(y_rectangle[0], y_rectangle[1])
                y_max = max(y_min+1, max(y_rectangle[0], y_rectangle[1]))

                X = np.random.randint(low=x_min, high=x_max,
                                      size=50).reshape((-1, 1))
                y = np.random.randint(low=y_min, high=y_max, size=50)

                X_plot = np.linspace(x_min, x_max, 100).reshape((-1, 1))
                degree = np.random.choice(self.degrees)
                model = make_pipeline(PolynomialFeatures(degree), Ridge())
                model.fit(X, y)
                hair = model.predict(X_plot)
                angle = np.random.rand()*2*np.pi
                x_, y_ = rotPt(X_plot.squeeze(), hair, angle)
                draw_points = (np.asarray(
                    [x_, y_]).T).astype(np.int32)
                b = background.copy()
                cv2.polylines(b, [
                              draw_points], False, (0, 0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

                alpha = np.random.rand()*.4

                background = cv2.addWeighted(
                    b, alpha, background, 1 - alpha, 0)
            return Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        else:
            return img


if __name__ == '__main__':

    IMAGE_PATH = '/home/romain/Projects/Kaggle/melanoma_classification/data/train_260/ISIC_0015719.jpg'
    image = Image.open(IMAGE_PATH)
    transformer = transforms.Compose([
        # MicroscopeAugmentation(),
        CVHairAugmentation(p=1),
    ])
    aug_img = transformer(image)
    # print(aug_img._size)
    aug_img.show()
