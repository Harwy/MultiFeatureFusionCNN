import os
import numpy as np
from pathlib import Path
import math
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as t
from PIL import Image
# import scipy.stats as stats  #该模块包含了所有的统计分析函数

DATA_CSV_PATH = "E:/Dataset/"
TRAIN_CSV_PATH = DATA_CSV_PATH + 'morph2-224_train.csv'
VALID_CSV_PATH = DATA_CSV_PATH + 'morph2-224_test.csv'
TEST_CSV_PATH = DATA_CSV_PATH + 'morph2-224_test.csv'
IMAGE_PATH = 'E:/Dataset/morph2-224'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class MorphDataloader(object):
    def __init__(self, csv_path, img_path, img_size=224, mode='train', data_mode='DLDL', num_classes=101):
        self.csv_path = Path(csv_path, f'morph2_{mode}.csv')
        self.image_path = Path(img_path)
        self.data_mode = data_mode
        self.num_classes = num_classes
        if mode == 'train':
            self.transform = t.Compose([
                # t.CenterCrop((140, 140)),
                # t.Resize((230, 230)),
                t.RandomHorizontalFlip(0.5),
                t.RandomGrayscale(0.5),
                t.RandomRotation(30),
                t.RandomCrop((img_size, img_size)),
                t.ColorJitter(
                    brightness=0.4, 
                    contrast=0.4, 
                    saturation=0.4
                    ),
                t.ToTensor(), 
                t.Normalize(mean=mean, std=std)
                ])
        else:
            self.transform = t.Compose([
                # t.CenterCrop((140, 140)),
                # t.Resize((230, 230)),
                t.CenterCrop((img_size, img_size)),
                t.ToTensor(),
                t.Normalize(mean=mean, std=std)
                ])

    def load_data(self, batch_size, shuffle=True, num_workers=0):
        dataset = MorphDataset(csv_path=self.csv_path,
                                img_dir=self.image_path,
                                transform=self.transform, 
                                data_mode=self.data_mode, 
                                num_classes=self.num_classes)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        return dataloader

class MorphDataset(Dataset):
    def __init__(self, csv_path, img_dir, data_mode, transform=None, num_classes=101) -> None:
        super().__init__()
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.x = df['image'].values
        if num_classes == 101:
            self.y = df['age'].values
        else:
            self.y = df['label'].values
        self.transform = transform
        self.num_classes = num_classes
        self.data_mode = data_mode
        
    def __getitem__(self, index: int):
        img = Image.open(os.path.join(self.img_dir, self.x[index]))
        if self.transform is not None:
            img = self.transform(img)
        age = self.y[index]
            
        if self.data_mode == 'dldl':
            levels = [normal_sampling(int(age), i) for i in range(self.num_classes)]
            levels = [i if i > 1e-10 else 1e-10 for i in levels]
            # levels[levels<1e-10] = 1e-10
            levels = torch.tensor(levels, dtype=torch.float32)
            levels = levels / levels.sum()
        else: # rank ordinal lrank
            levels = [1]*age + [0]*(self.num_classes - 1 - age)
            levels = torch.tensor(levels, dtype=torch.float32)

        return img, age, levels

    def __len__(self) -> int:
        return self.y.shape[0]


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def TrainDataloader(img_size=120, data_mode="DLDL", batch_size=32, num_classes=101, num_workers=0):
    morph = MorphDataloader(DATA_CSV_PATH, IMAGE_PATH, img_size, 'train', data_mode, num_classes)
    return morph.load_data(batch_size, True, num_workers)

def ValidDataloader(img_size=120, data_mode="DLDL", batch_size=32, num_classes=101, num_workers=0):
    morph = MorphDataloader(DATA_CSV_PATH, IMAGE_PATH, img_size, 'valid', data_mode, num_classes)
    return morph.load_data(batch_size, True, num_workers)

def TestDataloader(img_size=120, data_mode="DLDL", batch_size=32, num_classes=101, num_workers=0):
    morph = MorphDataloader(DATA_CSV_PATH, IMAGE_PATH, img_size, 'test', data_mode, num_classes)
    return morph.load_data(batch_size, False, num_workers)


