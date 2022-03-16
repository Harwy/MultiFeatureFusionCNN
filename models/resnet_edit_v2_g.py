# coding: utf-8

#############################################
# 在Resnet+coral基础上
# Gabor & 小波变换
# 借鉴DLDL的双损失函数方法
#############################################

# Imports
import math
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import ScatLayer
import argparse
from torch.optim import lr_scheduler
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image

torch.backends.cudnn.deterministic = True

import sys
sys.path.append('..')
import loss
import options
from utils import GaborConv2d


BASE_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV_PATH = "E:/Dataset/morph/"
TRAIN_CSV_PATH = DATA_CSV_PATH + 'morph2_train.csv'
VALID_CSV_PATH = DATA_CSV_PATH + 'morph2_valid.csv'
TEST_CSV_PATH = DATA_CSV_PATH + 'morph2_test.csv'
IMAGE_PATH = 'E:/Dataset/morph2-aligned'

# Argparse helper
def parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',
                        type=int,
                        default=0)

    parser.add_argument('--seed',
                        type=int,
                        default=1)

    parser.add_argument('--numworkers',
                        type=int,
                        default=0)


    parser.add_argument('--outpath',
                        type=str,
                        required=True)

    parser.add_argument('--imp_weight',
                        type=int,
                        default=0)
    args = parser.parse_args()
    return args


def task_importance_weights(label_array):
    uniq = torch.unique(label_array)  # 获取年龄编号序列 0-54
    num_examples = label_array.size(0)

    m = torch.zeros(uniq.shape[0])

    for i, t in enumerate(torch.arange(torch.min(uniq), torch.max(uniq))):
        m_k = torch.max(torch.tensor([label_array[label_array > t].size(0), 
                                      num_examples - label_array[label_array > t].size(0)]))
        m[i] = torch.sqrt(m_k.float())

    imp = m/torch.max(m)  # 归一化
    return imp


###################
# Dataset
###################

class Morph2Dataset(Dataset):
    """Custom Dataset for loading MORPH face images"""

    def __init__(self,
                 csv_path, img_dir, transform=None, num_classes=101, mode='label'):

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['image'].values
        assert(mode in ['age', 'label'])
        self.y = df[mode].values
        self.transform = transform
        self.num_classes = num_classes

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        age = self.y[index]
        # 正态分布权重标签
        label = [normal_sampling(int(age), i) for i in range(self.num_classes)]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = torch.Tensor(label)
        label = label / label.sum()  # fix 权重
        # ranking年龄标签
        # levels = [1]*age + [0]*(self.num_classes - 1 - age)
        # levels = torch.tensor(levels, dtype=torch.float32)
        return img, label, age
        # return img, label, levels, age  # image, 正态分布年龄权重, ranking, 年龄

    def __len__(self):
        return self.y.shape[0]

def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k-mean)**2/(2*std**2))#/(math.sqrt(2*math.pi)*std)

##########################
# MODEL
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            GaborConv2d(3, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(512, 1, bias=False)
        # self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes - 1).float())
        self.fc = nn.Linear(512, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):  # 3*120*120
        x = self.conv1(x)  # 64*60*60

        x = self.layer1(x)  # 64*60*60
        x = self.layer2(x)  # 128*30*30
        x = self.layer3(x)  # 256*15*15
        x = self.layer4(x)  # 512*8*8

        x = self.avgpool(x)  # 512*1*1
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = torch.sigmoid(logits)
        probas = F.normalize(probas, p=1, dim=1)
        # logits = logits + self.linear_1_bias
        # probas = torch.sigmoid(logits)
        return logits, probas


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[3, 3, 3, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model


###########################################
# Initialize Cost, Model, and Optimizer
###########################################
def cost_fn(logits, levels, imp):
    val = (-torch.sum((F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels))*imp,
           dim=1))
    return torch.mean(val)

def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0, 0, 0
    CA3, CA5 = 0., 0.
    rank = torch.Tensor([i for i in range(NUM_CLASSES)]).to(DEVICE)
    for i, inputs in enumerate(data_loader):
        (feature, _, age) = inputs  # img, label, age
        feature = feature.to(device)
        # label = label.to(device)
        age = age.to(device)

        logits, probas = model(feature)
        # _, predicted_labels = torch.max(probas, 1)
        pred = torch.sum(probas*rank, dim=1)
        num_examples += age.size(0)
        mae += torch.sum(torch.abs(pred - age))
        CA3 += torch.sum(torch.abs(pred - age) <= 3)
        CA5 += torch.sum(torch.abs(pred - age) <= 5)
        mse += torch.sum((pred - age)**2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    CA3 = CA3.double() / num_examples
    CA5 = CA5.double() / num_examples
    return mae, mse, CA3, CA5



if __name__ == "__main__":
    args = parsers()

    NUM_WORKERS = args.numworkers

    if args.cuda >= 0:
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    if args.seed == -1:
        RANDOM_SEED = None
    else:
        RANDOM_SEED = args.seed

    IMP_WEIGHT = args.imp_weight

    PATH = args.outpath
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    LOGFILE = os.path.join(PATH, 'training.log')
    TEST_PREDICTIONS = os.path.join(PATH, 'test_predictions.log')
    TEST_ALLPROBAS = os.path.join(PATH, 'test_allprobas.tensor')

    # Logging

    header = []

    header.append('PyTorch Version: %s' % torch.__version__)
    header.append('CUDA device available: %s' % torch.cuda.is_available())
    header.append('Using CUDA device: %s' % DEVICE)
    header.append('Random Seed: %s' % RANDOM_SEED)
    header.append('Task Importance Weight: %s' % IMP_WEIGHT)
    header.append('Output Path: %s' % PATH)
    header.append('Script: %s' % sys.argv[0])

    with open(LOGFILE, 'w') as f:
        for entry in header:
            print(entry)
            f.write('%s\n' % entry)
            f.flush()

    ##########################
    # SETTINGS
    ##########################

    # Hyperparameters
    learning_rate = options.LEARNING_RATE
    num_epochs = options.NUM_EPOCHS

    # Architecture
    NUM_CLASSES = options.NUM_CLASSES

    BATCH_SIZE = options.BATCH_SIZE
    GRAYSCALE = options.GRAYSCALE

    df = pd.read_csv(TRAIN_CSV_PATH)
    ages = df['age'].values
    # NUM_CLASSES = ages.max() - ages.min() + 1
    del df
    ages = torch.tensor(ages, dtype=torch.float)

    # Data-specific scheme
    if not IMP_WEIGHT:
        imp = torch.ones(NUM_CLASSES-1, dtype=torch.float)
    elif IMP_WEIGHT == 1:  # 年龄权重
        imp = task_importance_weights(ages)
        imp = imp[0:NUM_CLASSES-1]
    else:
        raise ValueError('Incorrect importance weight parameter.')
    imp = imp.to(DEVICE)

    custom_transform = transforms.Compose([transforms.CenterCrop((140, 140)),
                                       transforms.Resize((128, 128)),
                                       transforms.RandomCrop((120, 120)),
                                       transforms.ToTensor()])

    train_dataset = Morph2Dataset(csv_path=TRAIN_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                transform=custom_transform, 
                                mode="age")

    custom_transform2 = transforms.Compose([transforms.CenterCrop((140, 140)),
                                        transforms.Resize((128, 128)),
                                        transforms.CenterCrop((120, 120)),
                                        transforms.ToTensor()])

    test_dataset = Morph2Dataset(csv_path=TEST_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                transform=custom_transform2, 
                                mode="age")

    valid_dataset = Morph2Dataset(csv_path=VALID_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                transform=custom_transform2, 
                                mode="age")

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS)

    valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=NUM_WORKERS)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=NUM_WORKERS)

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    model = resnet34(NUM_CLASSES, GRAYSCALE)

    model.to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=options.WEIGHT_DECAY)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
                            weight_decay=options.WEIGHT_DECAY, 
                            momentum=options.MOMENTUM,)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=options.MILESTONES, gamma=options.GAMMA)
    # lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_ranking, milestones=options.MILESTONES, gamma=options.GAMMA)
    start_time = time.time()

    best_mae, best_rmse, best_epoch = 999, 999, -1
    best_CA3, best_CA5 = 100., 100.
    rank = torch.Tensor([i for i in range(101)]).to(DEVICE)
    for epoch in range(num_epochs):

        model.train()
        with tqdm(train_loader) as t:
            for batch_idx, inputs in enumerate(t):
                (feature, label, age) = inputs  # img, label, age
                feature = feature.to(DEVICE)
                label = label.to(DEVICE)
                age = age.to(DEVICE)

                # FORWARD AND BACK PROP
                logits, probas = model(feature)
                pred_age = torch.sum(probas*rank, dim=1)
                loss1 = loss.kl_loss(probas, label)
                loss2 = loss.L1_loss(pred_age, age)
                cost = loss1 + loss2
                optimizer.zero_grad()
                cost.backward()
                # UPDATE MODEL PARAMETERS
                optimizer.step()

                # LOGGING
                t.set_description('Epoch: %03d/%03d | Cost: %.4f'
                        % (epoch+1, num_epochs, cost))
                if not batch_idx % 50:
                    s = ('Epoch: %03d/%03d | Cost: %.4f'
                        % (epoch+1, num_epochs, cost))
                    # print(s)
                    with open(LOGFILE, 'a') as f:
                        f.write('%s\n' % s)
                
        lr_scheduler.step()

        model.eval()
        with torch.set_grad_enabled(False):
            valid_mae, valid_mse, valid_CA3, valid_CA5 = compute_mae_and_mse(model, valid_loader,
                                                    device=DEVICE)

        if valid_mae < best_mae:
            best_mae, best_rmse, best_epoch, best_CA3, best_CA5 = valid_mae, torch.sqrt(valid_mse), epoch, valid_CA3, valid_CA5
            ########## SAVE MODEL #############
            torch.save(model.state_dict(), os.path.join(PATH, 'best_model.pt'))


        s = 'MAE/RMSE/CA3/CA5: | Current Valid: %.4f/%.4f/%.4f/%.4f Ep. %d | Best Valid : %.4f/%.4f/%.4f/%.4f Ep. %d' % (
            valid_mae, torch.sqrt(valid_mse), valid_CA3, valid_CA5, epoch+1, 
            best_mae, best_rmse, best_CA3, best_CA5,  best_epoch+1)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

        s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

    ########## EVALUATE LAST MODEL ######
    model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference

        train_mae, train_mse, train_CA3, train_CA5 = compute_mae_and_mse(model, train_loader,
                                                device=DEVICE)
        valid_mae, valid_mse, valid_CA3, valid_CA5 = compute_mae_and_mse(model, valid_loader,
                                                device=DEVICE)
        test_mae, test_mse, test_CA3, test_CA5 = compute_mae_and_mse(model, test_loader,
                                                device=DEVICE)

        s = 'MAE/RMSE/CA3/CA5: | Train: %.4f/%.4f/%.4f/%.4f | Valid: %.4f/%.4f/%.4f/%.4f | Test: %.4f/%.4f/%.4f/%.4f' % (
            train_mae, torch.sqrt(train_mse), train_CA3, train_CA5, 
            valid_mae, torch.sqrt(valid_mse), valid_CA3, valid_CA5, 
            test_mae, torch.sqrt(test_mse), test_CA3, test_CA5)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

    s = 'Total Training Time: %.2f min' % ((time.time() - start_time)/60)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)


    ########## EVALUATE BEST MODEL ######
    model.load_state_dict(torch.load(os.path.join(PATH, 'best_model.pt')))
    model.eval()

    with torch.set_grad_enabled(False):
        train_mae, train_mse, train_CA3, train_CA5 = compute_mae_and_mse(model, train_loader,
                                                device=DEVICE)
        valid_mae, valid_mse, valid_CA3, valid_CA5 = compute_mae_and_mse(model, valid_loader,
                                                device=DEVICE)
        test_mae, test_mse, test_CA3, test_CA5 = compute_mae_and_mse(model, test_loader,
                                                device=DEVICE)

        s = 'MAE/RMSE/CA3/CA5: | Train: %.4f/%.4f/%.4f/%.4f | Valid: %.4f/%.4f/%.4f/%.4f | Test: %.4f/%.4f/%.4f/%.4f' % (
            train_mae, torch.sqrt(train_mse), train_CA3, train_CA5, 
            valid_mae, torch.sqrt(valid_mse), valid_CA3, valid_CA5, 
            test_mae, torch.sqrt(test_mse), test_CA3, test_CA5)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)


    ########## SAVE PREDICTIONS ######
    # print("########## SAVE PREDICTIONS ##########")
    # all_pred = []
    # all_probas = []
    # with torch.set_grad_enabled(False):
    #     with tqdm(test_loader) as t:
    #         for inputs in t:
    #             (features, _, _, age) = inputs
    #             features = features.to(DEVICE)
    #             _, probas, _, probas_ranking = model(features)
    #             all_probas.append(probas)
    #             predicted_labels = torch.sum(probas*rank, dim=1)
    #             predict_levels = probas_ranking > 0.5
    #             predicted_labels = torch.sum(predict_levels, dim=1)
    #             lst = [str(int(i)) for i in predicted_labels]
    #             all_pred.extend(lst)

    # torch.save(torch.cat(all_probas).to(torch.device('cpu')), TEST_ALLPROBAS)
    # with open(TEST_PREDICTIONS, 'w') as f:
    #     all_pred = ','.join(all_pred)
    #     f.write(all_pred)
print("########## 程序结束 ##########")