from PIL import Image, ImageFile
import numpy as np
import torch
from torchvision import transforms
import os
from data.augmentations import MicroscopeAugmentation, HairAugmentation
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MelanomaDataset:

    def __init__(self, image_paths, df, metafeatures, resize=None, resize_alg=Image.LANCZOS, augmentation=False, test=False):
        self.image_paths = image_paths
        self.df = df
        self.metafeatures = metafeatures
        self.resize = resize
        self.resize_alg = resize_alg
        self.augmentation = augmentation
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(os.path.join(
            self.image_paths, self.df.iloc[index]['image_name'] + '.jpg'))
        metadata = np.array(
            self.df.iloc[index][self.metafeatures].values, dtype=np.float32)
        target = self.df.iloc[index]['target']
        if self.resize:
            image = image.resize(self.resize, resample=self.resize_alg)
        if self.test:
            transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ])
        else:
            transformer = transforms.Compose([
                # transforms.RandomResizedCrop(size=260, scale=(0.7, 1.0)),
                HairAugmentation(p=.2),
                transforms.RandomHorizontalFlip(p=.5),
                transforms.RandomVerticalFlip(p=.5),
                transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
                # MicroscopeAugmentation(p=.2, size=(260, 260)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225])
            ])
        return {
            "image": transformer(image),
            "metadata": torch.tensor(metadata, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.long)
        }
