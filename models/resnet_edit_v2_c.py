# coding: utf-8

# Imports
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append("..")
import options
from datasets import MorphDataset
from torchvision import transforms
from torch.utils.data import DataLoader

description = "Resnet_lite + coral + EMD"

TEST = options.TEST
# BASE_ROOT = os.path.dirname(os.path.abspath(__file__))
if TEST is not True:
    DATA_CSV_PATH = "E:/Dataset/morph/"
    TRAIN_CSV_PATH = DATA_CSV_PATH + 'morph2_train.csv'
    VALID_CSV_PATH = DATA_CSV_PATH + 'morph2_valid.csv'
    TEST_CSV_PATH = DATA_CSV_PATH + 'morph2_test.csv'
    IMAGE_PATH = 'E:/Dataset/morph2-aligned'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DATA_CSV_PATH = "D:/datasets/morph/"
    TRAIN_CSV_PATH = DATA_CSV_PATH + 'morph2_train.csv'
    VALID_CSV_PATH = DATA_CSV_PATH + 'morph2_valid.csv'
    TEST_CSV_PATH = DATA_CSV_PATH + 'morph2_test.csv'
    IMAGE_PATH = 'D:/datasets/morph2-aligned'

    DEVICE = torch.device("cuda")
print("MODE in Test: ", TEST)
print("progress running device:", DEVICE)

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

##########################
# MODEL vgg16
##########################
class vgg16(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        net = models.vgg16(pretrained=True)  # 加载预训练参数的vgg16
        self.features = nn.Sequential(*list(net.children()))[:-1]
        self.classifier = nn.Sequential(
            nn.Linear(25088, 1024, bias=True), 
            nn.LeakyReLU(inplace=False),
            nn.Dropout(0.5, inplace=False), 
            nn.Linear(1024, 1024, bias=True), 
            nn.LeakyReLU(inplace=False),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(1024, num_classes-1),
        )
    
    def forward(self, x):
        x = self.features(x)  # (batch, 101)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probas = torch.sigmoid(logits)
        return logits, probas


###########################################
# Initialize Cost, Model, and Optimizer
###########################################
class EMDLoss(nn.Module):
    # 不要忘记继承Module
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, output, target):
        """output和target都是1-D张量,换句话说,每个样例的返回是一个标量.
        """
        output = torch.softmax(output, dim=1)
        target = torch.softmax(target, dim=1)
        loss = output - target
        
        for i in range(1, output.shape[1]):
            loss[:, i] += loss[:, i-1]
        # 不要忘记返回scalar
        loss = torch.pow(loss, 2)
        return torch.mean(loss)


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    
    def forward(self, output, target):
        val = (-torch.sum((F.logsigmoid(output)*target + (F.logsigmoid(output) - output)*(1-target)), dim=1))
        return torch.mean(val)

def compute_mae_and_mse(model, data_loader, device):
    mae, num_examples = 0, 0
    CA3, CA5 = 0., 0.
    for i, inputs in enumerate(data_loader):
        (features, age, _) = inputs  # img, label, age
        features = features.to(device)
        # label = label.to(device)
        age = age.to(device)

        _, probas = model(features)
        pred = torch.sum(probas > 0.5, dim=1)
        num_examples += age.size(0)
        mae += torch.sum(torch.abs(pred - age))
        CA3 += torch.sum(torch.abs(pred - age) <= 3)
        CA5 += torch.sum(torch.abs(pred - age) <= 5)
    mae = mae.float() / num_examples
    CA3 = CA3.double() / num_examples
    CA5 = CA5.double() / num_examples
    return mae, CA3, CA5



def train(model_, dataloaders_, optimizer_, lr_scheduler, num_epochs_, writers, PATH, LOGFILE):
    since = time.time()
    val_acc_history = []

    emd_loss = EMDLoss().to(DEVICE)
    kl_loss = KLLoss().to(DEVICE)

    best_loss = 100000.0
    best_mae, best_epoch = 999, -1
    best_CA3, best_CA5 = 0., 0.

    for epoch in range(num_epochs_):
        print('\nEpoch {}/{}'.format(epoch + 1, num_epochs_))
        print('-' * 10)

        for phase in dataloaders_.keys():
            if phase == 'train':
                model_.train()  # Set model to training mode
                print('in train mode...')
            else:
                print('in {} mode...'.format(phase))
                model_.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects_3 = 0.
            running_corrects_5 = 0.
            running_mae = 0.
            running_len = 0

            with tqdm(dataloaders_[phase]) as t:
                for inputs in t:
                    (features, ages, levels) = inputs
                    features = features.to(DEVICE)
                    ages = ages.to(DEVICE)
                    levels = levels.to(DEVICE)

                    # zero the parameter gradients
                    optimizer_.zero_grad()

                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        logits, probas = model_(features)
                        pred = torch.sum(probas > 0.5, dim=1)
                        loss = kl_loss(logits, levels) #+ 0.25*emd_loss(logits, levels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer_.step()

                    # statistics
                    running_len += features.size(0)
                    running_loss += loss.item() * features.size(0)
                    running_corrects_3 += torch.sum(torch.abs(pred - ages) < 3)  # CA 3
                    running_corrects_5 += torch.sum(torch.abs(pred - ages) < 5)  # CA 5
                    running_mae += torch.sum(torch.abs(pred - ages))
                    # tqdm description
                    t.set_description('loss={loss:.4f} CA3={CA3:.4f} CA5={CA5:.4f} MAE={mae:.4f}'.format(
                                    loss=running_loss/running_len, 
                                    CA3=running_corrects_3/running_len, 
                                    CA5=running_corrects_5/running_len, 
                                    mae=running_mae/running_len))

            epoch_loss = running_loss / len(dataloaders_[phase].dataset)
            CA_3 = running_corrects_3.double() / len(dataloaders_[phase].dataset)
            CA_5 = running_corrects_5.double() / len(dataloaders_[phase].dataset)
            MAE = running_mae.double() / len(dataloaders_[phase].dataset)

            writers.add_scalar(f'{phase}/loss', epoch_loss, global_step=epoch)
            writers.add_scalar(f'{phase}/MAE', MAE.item(), global_step=epoch)
            writers.add_scalar(f'{phase}/CA_3', CA_3.item(), global_step=epoch)
            writers.add_scalar(f'{phase}/CA_5', CA_5.item(), global_step=epoch)

            s = '{} Loss: {:.4f} CA_3: {:.4f}, CA_5: {:.4f} MAE:{:.4f}'.format(phase, epoch_loss, CA_3, CA_5, MAE)
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)
            time_elapsed = time.time() - since
            s = 'Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)

            # deep copy the model
            if phase == 'valid' :
                val_acc_history.append(MAE)
                if MAE < best_mae:
                    best_loss = epoch_loss
                    best_mae, best_epoch, best_CA3, best_CA5 = MAE, epoch, CA_3, CA_5
                    ########## SAVE MODEL #############
                    torch.save(model_.state_dict(), str(PATH.joinpath('best_model.pt')))
                s = ('MAE/CA3/CA5: | Current Valid: %.4f/%.4f/%.4f Ep. %d | Best Valid : %.4f/%.4f/%.4f Ep. %d' % (
                    MAE, CA_3, CA_5, epoch+1,
                    best_mae, best_CA3, best_CA5,  best_epoch+1
                ))
                print(s)
                with open(LOGFILE, 'a') as f:
                    f.write('%s\n' % s)
        lr_scheduler.step() 

    time_elapsed = time.time() - since
    s = ('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)
    print('Best val MAE: {:4f}'.format(best_loss))
    
    return model_, val_acc_history


def main():
    ##########################
    # SETTINGS
    ##########################
    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 72  # 80 

    # Architecture
    NUM_CLASSES = 101

    if TEST is not True:
        BATCH_SIZE = 64
    else:
        BATCH_SIZE = 1
    GRAYSCALE = options.GRAYSCALE

    data_mode = 'rank'

    args = parsers()

    NUM_WORKERS = args.numworkers

    if args.seed == -1:
        RANDOM_SEED = None
    else:
        RANDOM_SEED = args.seed

    IMP_WEIGHT = args.imp_weight

    PATH = Path(args.outpath)
    if PATH.exists(): 
        import shutil
        shutil.rmtree(PATH)
    PATH.mkdir(parents=True, exist_ok=True)
    LOGFILE = str(PATH.joinpath('training.log'))
    writers = SummaryWriter(str(PATH.joinpath('runs')))

    # Logging

    header = []

    header.append('PyTorch Version: %s' % torch.__version__)
    header.append('CUDA device available: %s' % torch.cuda.is_available())
    header.append('Using CUDA device: %s' % DEVICE)
    header.append('Random Seed: %s' % RANDOM_SEED)
    header.append('Task Importance Weight: %s' % IMP_WEIGHT)
    header.append('Output Path: %s' % PATH)
    header.append('Script: %s' % sys.argv[0])
    header.append('Learning Rate: {}'.format(learning_rate))
    header.append('Num Epochs: {}'.format(num_epochs))
    header.append('Num Classes: {}'.format(NUM_CLASSES))
    header.append('Batch Size: {}'.format(BATCH_SIZE))
    header.append('Data Mode: {}'.format(data_mode))
    header.append('Description: ' + description)  # 所用方法
    

    with open(LOGFILE, 'w') as f:
        for entry in header:
            print(entry)
            f.write('%s\n' % entry)
            f.flush()

    ##########################
    # data loader
    ##########################
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    custom_transform = transforms.Compose([
        transforms.CenterCrop((140, 140)),
        transforms.Resize((230, 230)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)])

    train_dataset = MorphDataset(csv_path=TRAIN_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                num_classes=NUM_CLASSES,
                                transform=custom_transform, 
                                data_mode=data_mode)

    custom_transform2 = transforms.Compose([
        transforms.CenterCrop((140, 140)),
        transforms.Resize((230, 230)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)])

    test_dataset = MorphDataset(csv_path=TEST_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                num_classes=NUM_CLASSES,
                                transform=custom_transform2, 
                                data_mode=data_mode)

    valid_dataset = MorphDataset(csv_path=VALID_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                num_classes=NUM_CLASSES,
                                transform=custom_transform2, 
                                data_mode=data_mode)

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

    loader = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader,
    }

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    if RANDOM_SEED == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = vgg16(NUM_CLASSES)

    """
    start to train and test
    """
    model.to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
    #                         weight_decay=options.WEIGHT_DECAY, 
    #                         momentum=options.MOMENTUM,)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=options.MILESTONES, gamma=options.GAMMA)

    model, hist = train(model, loader, optimizer, lr_scheduler, num_epochs, writers, PATH, LOGFILE)

    ########## EVALUATE LAST MODEL ######
    model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference

        train_mae, train_CA3, train_CA5 = compute_mae_and_mse(model, train_loader,
                                                device=DEVICE)
        valid_mae, valid_CA3, valid_CA5 = compute_mae_and_mse(model, valid_loader,
                                                device=DEVICE)
        test_mae, test_CA3, test_CA5 = compute_mae_and_mse(model, test_loader,
                                                device=DEVICE)

        s = 'LAST | MAE/CA3/CA5: | Train: %.4f/%.4f/%.4f | Valid: %.4f/%.4f/%.4f | Test: %.4f/%.4f/%.4f' % (
            train_mae, train_CA3, train_CA5, 
            valid_mae, valid_CA3, valid_CA5, 
            test_mae, test_CA3, test_CA5)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)


    ########## EVALUATE BEST MODEL ######
    model.load_state_dict(torch.load(str(PATH.joinpath('best_model.pt'))))
    model.eval()

    with torch.set_grad_enabled(False):
        train_mae, train_CA3, train_CA5 = compute_mae_and_mse(model, train_loader,
                                                device=DEVICE)
        valid_mae, valid_CA3, valid_CA5 = compute_mae_and_mse(model, valid_loader,
                                                device=DEVICE)
        test_mae, test_CA3, test_CA5 = compute_mae_and_mse(model, test_loader,
                                                device=DEVICE)

        s = 'BEST | MAE/CA3/CA5: | Train: %.4f/%.4f/%.4f | Valid: %.4f/%.4f/%.4f | Test: %.4f/%.4f/%.4f' % (
            train_mae, train_CA3, train_CA5, 
            valid_mae, valid_CA3, valid_CA5, 
            test_mae, test_CA3, test_CA5)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

    print("==== save model ====")
    torch.save(model, str(PATH.joinpath('resnet_lite_coral_EMD_model_adam.pth')))

    print("########## 程序结束 ##########")


if __name__ == "__main__":
    main()

