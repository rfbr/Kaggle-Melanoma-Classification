import os

TRAIN_DATA_PATH = os.path.abspath('../data/train.csv')
TEST_DATA_PATH = os.path.abspath('../data/test.csv')
SUB_PATH = os.path.abspath('../data/sample_submission.csv')
TRAIN_IMAGE_PATH = os.path.abspath('../data/jpeg/train')
TEST_IMAGE_PATH = os.path.abspath('../data/jpeg/test')
SIZE = 256
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1.5e-5
LEARNING_RATE_META = 1e-5
