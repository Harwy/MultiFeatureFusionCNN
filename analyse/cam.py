import torch
from torch import nn
from torch.nn import functional as F
from skimage import io
import numpy as np
import cv2
import sys
sys.path.append(r"E:/FeatureAgeNet")


class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        pred = torch.sum(torch.sigmoid(output)>0.5, axis=1)
        # print("pred: ", pred)
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        # target = output[0][index]
        # print("target: ", target.shape)
        levels = [1]*index + [0]*(100 - index)
        levels = torch.tensor(levels, dtype=torch.float32).cuda()
        val = -torch.sum(F.logsigmoid(output)*levels + (F.logsigmoid(output) - output)*(1-levels), dim=1)
        target = torch.mean(val)
        target.backward()
        
        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (224, 224))
        return cam, pred.cpu().data.numpy()

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

def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

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
    import pandas as pd
    model = torch.load(r"E:\FeatureAgeNet\Logs\resnet_rank_224_224_size_90_epoches\best_all_model.pth")
    # model = model.cpu()
    model.eval()

    df = pd.read_csv(r"E:\Dataset\morph2-224.csv")
    root = "E:/Dataset/morph2-224/"
    # name = r"E:\Dataset\morph2-224\338107_01F17.jpg"
    for name in df.sample(5)['path'].values:
        image_name = name.split("\\")[-1]
        img = io.imread(root + name)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        inputs = prepare_input(img)

        # 输出图像
        image_dict = {}
        # Grad-CAM
        layer_name = get_last_conv_name(model)
        # layer_name
        grad_cam = GradCAM(model, layer_name)
        mask, pred = grad_cam(inputs.cuda(), int(image_name[-6:-4]))  # cam mask
        image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
        io.imsave(f"./cam/{image_name}_{pred}.jpg", image_dict['cam'])
    