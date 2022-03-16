import cv2
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


import sys
sys.path.append(r"E:/FeatureAgeNet")

def get_img(url):
    img = cv2.imread(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (np.float32(img) / 255.0 - mean) / std
    x = x.transpose((2, 0, 1))
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x)
    x = torch.tensor(x, dtype=torch.float)
    return img, x

