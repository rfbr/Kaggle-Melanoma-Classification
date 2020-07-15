import cv2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from PIL import Image
import io
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    IMAGE_PATH = '/home/romain/Projects/Kaggle/melanoma_classification/data/train_224/ISIC_0015719.jpg'
    test_image = Image.open(IMAGE_PATH)
    n_hair = np.random.randint(30, 100)
    hair_aug = HairAugmentation(p=1)
    test = hair_aug(test_image)
    test.show()
# degrees =
# img_shape = test_image._size
# plt.imshow(test_image)

# for _ in range(n_hair):
#     # Random rectangle
#     x_rectangle = np.random.randint(low=0, high=img_shape[0]-1, size=2)
#     y_rectangle = np.random.randint(low=0, high=img_shape[1]-1, size=2)

#     x_min = min(x_rectangle[0], x_rectangle[1])
#     x_max = max(x_min+1, max(x_rectangle[0], x_rectangle[1]))
#     y_min = min(y_rectangle[0], y_rectangle[1])
#     y_max = max(y_min+1, max(y_rectangle[0], y_rectangle[1]))

#     X = np.random.randint(low=x_min, high=x_max, size=50).reshape((-1, 1))
#     y = np.random.randint(low=y_min, high=y_max, size=50)

#     X_plot = np.linspace(x_min, x_max, 100).reshape((-1, 1))
#     degree = np.random.choice(degrees)
#     model = make_pipeline(PolynomialFeatures(degree), Ridge())
#     model.fit(X, y)
#     hair = model.predict(X_plot)

#     # first of all, the base transformation of the data points is needed
#     base = plt.gca().transData
#     random_degree = np.random.randint(0, 360)
#     rot = transforms.Affine2D().rotate_deg_around(
#         img_shape[0]//2, img_shape[1]//2, random_degree)
#     alpha = np.random.rand()*.4
#     plt.plot(X_plot, hair, 'black', alpha=alpha, transform=base+rot)
# buf = io.BytesIO()
# plt.savefig(buf, format='png')
# im = Image.open(buf)
# buf.close()
