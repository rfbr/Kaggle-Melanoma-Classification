from PIL import Image, ImageEnhance
IMAGE_PATH = '/home/romain/Projects/Kaggle/melanoma_classification/data/train_224/ISIC_0015719.jpg'
image = Image.open(IMAGE_PATH)
image.show()
# image brightness enhancer
enhancer = ImageEnhance.Contrast(image)
factor = 1.5  # increase contrast
im_output = enhancer.enhance(factor)
im_output.show()
