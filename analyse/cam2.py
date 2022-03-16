import numpy as np
import cv2
import torch
from torch import nn
import pandas as pd
from skimage import io
import sys
sys.path.append(r"E:/FeatureAgeNet")
from matplotlib import pyplot as plt

# 中间特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = []
        # print(self.submodule._modules.items())
        for name, module in self.submodule._modules.items():
            if "fc" in name: 
                # print(name)
                x = x.view(x.size(0), -1)
            # print(module)
            x = module(x)
            # print(name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

def acc_output(a):
    for i in range(7):
        a[0][i] = 0
        a[i][0] = 0
        a[6][i] = 0
        a[i][6] = 0
    for i in [0,6]:
        for j in [0,6]:
            a[i][j] = 0
    return a

def prepare_input(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)

def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)

if __name__ == "__main__":
    model = torch.load(r"E:\FeatureAgeNet\Logs\resnet_rank_224_224_size_90_epoches\best_all_model.pth")
    model = model.cpu()
    model.eval()
    exact_list = ["conv1","layer1", "layer2", "layer3", "layer4", "downsample"]
    myexactor = FeatureExtractor(model, exact_list)

    root = "E:/Dataset/morph2-224/"
    df = pd.read_csv(r"E:\Dataset\morph2-224.csv")
    df = df[df.age == 18]

    for name in df.sample(5)['path'].values:
        # name = r"E:\Dataset\morph2-224\009055_1M54.jpg"
        image_name = name.split("\\")[-1]
        img = io.imread(root + name)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        inputs = prepare_input(img)

        pred = model(inputs)
        pred = torch.sum(torch.sigmoid(pred)>0.5, axis=1).data.numpy()
        y = myexactor(inputs)
        a = y[-2].cpu().data.numpy()[0, :, :, :]  # (512, 7, 7)
        a = np.sum(a, axis=0) # (7, 7)
        a = np.maximum(0, a)  # relu

        a -= a.min()
        a /= a.max()  # normalize
        a = acc_output(a)
        
        # plt.imshow(a, cmap="jet")
        a = cv2.resize(a, (224,224))
        # mask转为heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * a), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]  # gbr to rgb
        # 合并heatmap到原始图像
        cam = heatmap + np.float32(img)
        img = norm_image(cam)
        io.imsave(f"./cam2/{image_name}_{pred}.jpg", img)
    # plt.imshow(norm_image(cam))
    # plt.show()
