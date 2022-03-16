# coding: utf-8

#############################################
# Ordinal Regression Code with ResNet-34
#############################################

# Imports

import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import ScatLayer
from utils import GaborConv2d
import argparse
import sys
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image

torch.backends.cudnn.deterministic = True


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
                 csv_path, img_dir, transform=None, num_classes=55, mode='label'):

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

        label = self.y[index]
        levels = [1]*label + [0]*(self.num_classes - 1 - label)  # 编码
        levels = torch.tensor(levels, dtype=torch.float32)

        return img, label, levels  # image, 年龄， 年龄序列

    def __len__(self):
        return self.y.shape[0]


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
        self.conv1 = nn.Conv2d(in_dim, 23, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_gabor = GaborConv2d(in_dim, 20, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_scat = ScatLayer()  # 21
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, (self.num_classes-1)*2)

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

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_gabor(x)
        x3 = self.conv1_scat(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)  # 二分类器数*2  (batch, 54*2)
        logits = logits.view(-1, (self.num_classes-1), 2) # (batch, 54, 2)
        probas = F.softmax(logits, dim=2)[:, :, 1]
        return logits, probas


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model


###########################################
# Initialize Cost, Model, and Optimizer
###########################################

def cost_fn(logits, levels, imp):
    val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1]*levels
                      + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels))*imp, dim=1))
    return torch.mean(val)



def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0, 0, 0
    CA3, CA5 = 0., 0.
    for i, (features, targets, levels) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)  # age
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        CA3 += torch.sum(torch.abs(predicted_labels - targets) <= 3)
        CA5 += torch.sum(torch.abs(predicted_labels - targets) <= 5)
        mse += torch.sum((predicted_labels - targets)**2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    CA3 = CA3.double() / num_examples
    CA5 = CA5.double() / num_examples
    return mae, mse, CA3, CA5



if __name__ == "__main__":
    # freeze_support()
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
    learning_rate = 0.0005
    num_epochs = 80

    # Architecture
    NUM_CLASSES = 62 #55

    BATCH_SIZE = 20
    GRAYSCALE = False

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
                                num_classes=NUM_CLASSES)

    custom_transform2 = transforms.Compose([transforms.CenterCrop((140, 140)),
                                        transforms.Resize((128, 128)),
                                        transforms.CenterCrop((120, 120)),
                                        transforms.ToTensor()])

    test_dataset = Morph2Dataset(csv_path=TEST_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                transform=custom_transform2, 
                                num_classes=NUM_CLASSES)

    valid_dataset = Morph2Dataset(csv_path=VALID_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                transform=custom_transform2, 
                                num_classes=NUM_CLASSES)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()

    best_mae, best_rmse, best_epoch = 999, 999, -1
    CA3, CA5 = 100., 100.
    for epoch in range(num_epochs):

        model.train()
        with tqdm(train_loader) as t:
            for batch_idx, (features, targets, levels) in enumerate(t):

                features = features.to(DEVICE)
                # targets = targets
                targets = targets.to(DEVICE)
                levels = levels.to(DEVICE)

                # FORWARD AND BACK PROP
                logits, probas = model(features)
                cost = cost_fn(logits, levels, imp)
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

        model.eval()
        with torch.set_grad_enabled(False):
            valid_mae, valid_mse, valid_CA3, valid_CA5 = compute_mae_and_mse(model, valid_loader,
                                                    device=DEVICE)

        if valid_mae < best_mae:
            best_mae, best_rmse, best_epoch, best_CA3, best_CA5 = valid_mae, torch.sqrt(valid_mse), epoch, valid_CA3, valid_CA5
            ########## SAVE MODEL #############
            torch.save(model.state_dict(), os.path.join(PATH, 'best_model.pt'))


        s = 'MAE/RMSE/CA3/CA5: | Current Valid: %.2f/%.2f/%.2f/%.2f Ep. %d | Best Valid : %.2f/%.2f/%.2f/%.2f Ep. %d' % (
            valid_mae, torch.sqrt(valid_mse), valid_CA3, valid_CA5, epoch, 
            best_mae, best_rmse, best_CA3, best_CA5,  best_epoch)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

        s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

    model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference

        train_mae, train_mse, train_CA3, train_CA5 = compute_mae_and_mse(model, train_loader,
                                                device=DEVICE)
        valid_mae, valid_mse, valid_CA3, valid_CA5 = compute_mae_and_mse(model, valid_loader,
                                                device=DEVICE)
        test_mae, test_mse, test_CA3, test_CA5 = compute_mae_and_mse(model, test_loader,
                                                device=DEVICE)

        s = 'MAE/RMSE/CA3/CA5: | Train: %.2f/%.2f/%.2f/%.2f | Valid: %.2f/%.2f/%.2f/%.2f | Test: %.2f/%.2f/%.2f/%.2f' % (
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

        s = 'MAE/RMSE/CA3/CA5: | Train: %.2f/%.2f/%.2f/%.2f | Valid: %.2f/%.2f/%.2f/%.2f | Test: %.2f/%.2f/%.2f/%.2f' % (
            train_mae, torch.sqrt(train_mse), train_CA3, train_CA5, 
            valid_mae, torch.sqrt(valid_mse), valid_CA3, valid_CA5, 
            test_mae, torch.sqrt(test_mse), test_CA3, test_CA5)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)


    ########## SAVE PREDICTIONS ######
    all_pred = []
    all_probas = []
    with torch.set_grad_enabled(False):
        with tqdm(test_loader) as t:
            for (features, targets, levels) in t:

                features = features.to(DEVICE)
                logits, probas = model(features)
                all_probas.append(probas)
                predict_levels = probas > 0.5
                predicted_labels = torch.sum(predict_levels, dim=1)
                lst = [str(int(i)) for i in predicted_labels]
                all_pred.extend(lst)

    torch.save(torch.cat(all_probas).to(torch.device('cpu')), TEST_ALLPROBAS)
    with open(TEST_PREDICTIONS, 'w') as f:
        all_pred = ','.join(all_pred)
        f.write(all_pred)