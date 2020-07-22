from PIL import Image, ImageFile
import numpy as np
import torch
from torchvision import transforms
import os
from data.augmentations import MicroscopeAugmentation, CVHairAugmentation, CutOut
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
            h_transformer = transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor(),
                # transforms.Lambda(lambda img: img * 2.0 - 1.0)
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ])
            v_transformer = transforms.Compose([
                transforms.RandomVerticalFlip(p=1),
                transforms.ToTensor(),
                # transforms.Lambda(lambda img: img * 2.0 - 1.0)
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ])
            # rc_transformer = transforms.Compose([
            #     transforms.RandomResizedCrop(size=256, scale=(
            #         0.5*(1 + np.random.rand()), 0.5*(1 + np.random.rand())), interpolation=Image.LANCZOS),
            #     transforms.ToTensor(),
            #     # transforms.Lambda(lambda img: img * 2.0 - 1.0)
            #     transforms.Normalize([0.485, 0.456, 0.406], [
            #                          0.229, 0.224, 0.225]),
            # ])
            n_transformer = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Lambda(lambda img: img * 2.0 - 1.0)
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ])
            cj_transformer = transforms.Compose([
                transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
                transforms.ToTensor(),
                # transforms.Lambda(lambda img: img * 2.0 - 1.0)
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ])
            # rot_transformer = transforms.Compose([
            #     transforms.RandomChoice([
            #         transforms.RandomChoice([
            #             transforms.RandomAffine(
            #                 180, scale=(.8, 1.2), shear=10, resample=Image.NEAREST),
            #             transforms.RandomAffine(
            #                 180, scale=(.8, 1.2), shear=10, resample=Image.BICUBIC),
            #             transforms.RandomAffine(
            #                 180, scale=(.8, 1.2), shear=10, resample=Image.BILINEAR)
            #         ]),
            #         transforms.RandomChoice([
            #             transforms.RandomRotation(
            #                 degrees=180, resample=Image.NEAREST),
            #             transforms.RandomRotation(
            #                 degrees=180, resample=Image.BICUBIC),
            #             transforms.RandomRotation(
            #                 degrees=180, resample=Image.BILINEAR)
            #         ])
            #     ]),
            #     transforms.ToTensor(),
            #     # transforms.Lambda(lambda img: img * 2.0 - 1.0)
            #     transforms.Normalize([0.485, 0.456, 0.406], [
            #         0.229, 0.224, 0.225])
            # ])

            return {
                "image": [n_transformer(image), h_transformer(image), v_transformer(image), cj_transformer(image)],
                "metadata": torch.tensor(metadata, dtype=torch.float32),
                "target": torch.tensor(target, dtype=torch.long)
            }
        else:
            transformer = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=image._size[0], scale=(.5*(1+np.random.rand()), .5*(1+np.random.rand())), interpolation=Image.LANCZOS),
                transforms.RandomHorizontalFlip(p=.5),
                transforms.RandomVerticalFlip(p=.5),
                # transforms.RandomChoice([
                #     transforms.RandomChoice([
                #         transforms.RandomAffine(
                #             180, scale=(.8, 1.2), shear=10, resample=Image.NEAREST),
                #         transforms.RandomAffine(
                #             180, scale=(.8, 1.2), shear=10, resample=Image.BICUBIC),
                #         transforms.RandomAffine(
                #             180, scale=(.8, 1.2), shear=10, resample=Image.BILINEAR)
                #     ]),
                #     transforms.RandomChoice([
                #         transforms.RandomRotation(
                #             degrees=180, resample=Image.NEAREST),
                #         transforms.RandomRotation(
                #             degrees=180, resample=Image.BICUBIC),
                #         transforms.RandomRotation(
                #             degrees=180, resample=Image.BILINEAR)
                #     ])
                # ]),
                transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
                CutOut(n_holes=1, length=16),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225]),
                # transforms.Lambda(lambda img: img * 2.0 - 1.0),
            ])
            return {
                "image": transformer(image),
                "metadata": torch.tensor(metadata, dtype=torch.float32),
                "target": torch.tensor(target, dtype=torch.long)
            }
