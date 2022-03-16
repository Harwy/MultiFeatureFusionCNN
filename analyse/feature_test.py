import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

import sys
sys.path.append(r"E:/FeatureAgeNet")

def get_input(path):
    # image = r"E:\Dataset\morph2-224\009055_1M54.jpg"
    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (np.float32(img) / 255.0 - mean) / std
    img = img.transpose((2, 0, 1))
    x = np.expand_dims(img, axis=0)
    x = torch.from_numpy(x)
    x = torch.tensor(x, dtype=torch.float)
    x = x.cuda()
    return x

if __name__ == "__main__":
    pass
    # model = torch.load(r"E:\FeatureAgeNet\Logs\resnet_rank_224_224_size_90_epoches\best_all_model.pth")
    # root = "E:/Dataset/morph2-224/"
    # df = pd.read_csv('E:/Dataset/morph2-224_test.csv')
    # df = df[df.age==16]
    # sample = df.sample(1)
    # sample.
    # x = get_input(path)
    # y =model(x)