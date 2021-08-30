import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from albumentations import *
from albumentations.pytorch import ToTensorV2


def get_transforms(need=('train', 'val'), img_size=(512, 384), mean=(0.55800916, 0.51224077, 0.47767341), std=(0.21817792, 0.23804603, 0.25183411)):

    transformations = {}
    if 'train' in need:
        transformations['train'] = Compose([
            Resize(img_size[0], img_size[1], p=1.0),
#             HorizontalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
#             HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
#             RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            GaussNoise(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
        transformations['val'] = Compose([
            Resize(img_size[0], img_size[1]),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    return transformations


class MaskDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.df = pd.read_csv(os.path.join(img_paths, 'train.csv'))
        self.img_paths = self.find_path(self.df, os.path.join(img_paths, 'images'))
        self.transform = transform
    
    def find_path(self, df,image_dir):
        file_path = []
        for img_path in df.path:
            for img in os.listdir(os.path.join(image_dir, img_path)):
                if (img.split('.')[0] != '') and (img.split('.')[-1] != 'ipynb_checkpoints'):
                    file_path.append(os.path.join(image_dir, img_path, img))
        return file_path
    
    def make_label(self,img_paths):
        gender = img_paths.split('/')[-2].split('_')[1]
        age = int(img_paths.split('/')[-2].split('_')[-1])

        if img_paths.split('/')[-1].split('.')[0] not in ['incorrect_mask','normal']:
            wear = 'mask'
        else:
            wear = img_paths.split('/')[-1].split('.')[0]

        gender_class = 0 if gender=='male' else 1
        age_class = int(age/30)
        if wear=='mask':
            wear_class = 0
        elif wear=='incorrect_mask':
            wear_class = 1
        else:
            wear_class = 2

        return 6*wear_class + 3*gender_class + age_class
    
    def set_transform(self, transform):
        """
        transform 함수를 설정하는 함수입니다.
        """
        self.transform = transform
    
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        label = self.make_label(self.img_paths[index])
        
        if self.transform:
            image = self.transform(image=np.array(image))
            
        return image, label

    def __len__(self):
        return len(self.img_paths)