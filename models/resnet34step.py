import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

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
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.downsample = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).float())

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
        x = self.conv1(x)  # 32*30*30

        x = self.layer1(x)  # 32*30*30
        x = self.layer2(x)  # 64*15*15
        x = self.layer3(x)  # 128*8*8
        x = self.layer4(x)  # 256*4*4
        x = self.downsample(x)  # 256*1*1
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas

def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model


def cost_fn(logits, levels):
    val = -torch.sum(F.logsigmoid(logits)*levels + (F.logsigmoid(logits) - logits)*(1-levels), dim=1)
    return torch.mean(val)


def test(model, data_loader, device, logger, mode):
    mae, num_examples = 0, 0
    CA = [0.]*10
    with tqdm(data_loader) as e:
        for inputs in e:
            (feature, age, _) = inputs
            feature = feature.to(device)
            age = age.to(device)

            _, probas = model(feature)
            pred = torch.sum(probas > 0.5, dim=1)
            num_examples += age.size(0)
            mae += torch.sum(torch.abs(pred - age))
            for i in range(len(CA)):
                CA[i] += torch.sum(torch.abs(pred - age) <= (i+1))
    mae = mae.float() / num_examples
    for i in range(len(CA)):
        CA[i] = CA[i].double() / num_examples
    logger.info("mode: {} | MAE: {:.4f} | CA: {} |".format(mode, mae, "/".join("%.4f" % i for i in CA)))
    return 


def train(model_, dataloaders_, optimizer_, lr_scheduler_, num_epochs_, writers, logger, PATH, device):
    val_acc_history = []

    best_loss = 100000.0
    best_mae, best_epoch = 999, -1
    best_CA3, best_CA5 = 0., 0.

    for epoch in range(num_epochs_):
        logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs_))
        logger.info('-' * 10)

        for phase in dataloaders_.keys():
            if phase == 'train':
                model_.train()
            else:
                model_.eval()
            logger.debug('in {} mode...'.format(phase))

            running_loss = 0.0
            running_corrects_3 = 0.
            running_corrects_5 = 0.
            running_mae = 0.
            running_len = 0

            with tqdm(dataloaders_[phase]) as t:
                for inputs in t:
                    (features, ages, levels) = inputs
                    features = features.to(device)
                    ages = ages.to(device)
                    levels = levels.to(device)

                    optimizer_.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        logits, probas = model_(features)
                        pred = torch.sum(probas > 0.5, dim=1)
                        loss = cost_fn(logits, levels)

                        if phase == 'train':
                            loss.backward()
                            optimizer_.step()

                    running_len += features.size(0)
                    running_loss += loss.item() * features.size(0)
                    running_corrects_3 += torch.sum(torch.abs(pred - ages) < 3)  # CA 3
                    running_corrects_5 += torch.sum(torch.abs(pred - ages) < 5)  # CA 5
                    running_mae += torch.sum(torch.abs(pred - ages))
                    # tqdm description
                    t.set_description('{mode} loss={loss:.4f} CA3={CA3:.4f} CA5={CA5:.4f} MAE={mae:.4f}'.format(
                                    mode=phase,
                                    loss=running_loss/running_len, 
                                    CA3=running_corrects_3/running_len, 
                                    CA5=running_corrects_5/running_len, 
                                    mae=running_mae/running_len))
            epoch_loss = running_loss / len(dataloaders_[phase].dataset)
            CA_3 = running_corrects_3.double() / len(dataloaders_[phase].dataset)
            CA_5 = running_corrects_5.double() / len(dataloaders_[phase].dataset)
            MAE = running_mae.double() / len(dataloaders_[phase].dataset)

            # writers[phase].add_scalars('data/loss', {phase: epoch_loss}, global_step=epoch)
            writers.add_scalar(f'{phase}/loss', epoch_loss, global_step=epoch)
            writers.add_scalar(f'{phase}/MAE', MAE.item(), global_step=epoch)
            writers.add_scalars(f'{phase}/CA', {'CA3': CA_3.item(), 'CA5': CA_5.item()}, global_step=epoch)

            logger.debug('{} Loss: {:.4f} CA_3: {:.4f}, CA_5: {:.4f} MAE:{:.4f}'.format(phase, epoch_loss, CA_3, CA_5, MAE))

            if phase == 'valid':
                val_acc_history.append(CA_3.item())
                if MAE < best_mae:
                    best_loss = epoch_loss
                    best_mae, best_epoch, best_CA3, best_CA5 = MAE, epoch, CA_3, CA_5
                    ##### save model #####
                    torch.save(model_.state_dict(), str(PATH.joinpath('best_model.pt')))
                logger.warning('MAE/CA3/CA5: | Current Valid: %.4f/%.4f/%.4f Ep. %d | Best Valid : %.4f/%.4f/%.4f Ep. %d' % (
                    MAE, CA_3, CA_5, epoch+1,
                    best_mae, best_CA3, best_CA5,  best_epoch+1
                ))
        lr_scheduler_.step()
    logger.info('Best val MAE: {:4f}'.format(best_loss))
    return model_, val_acc_history
