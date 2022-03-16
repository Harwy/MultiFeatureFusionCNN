from torch import nn
from torch.nn import functional as F
import torch
from torchvision.models.utils import load_state_dict_from_url
from tqdm import tqdm


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, SE=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.SE = SE
        if self.SE:
            self.SELayer = SELayer(self.oup, self.oup)

    def forward(self, x):
        if self.use_res_connect:
            x = x + self.conv(x)
        else:
            x =  self.conv(x)
        if self.SE:
            x = self.SELayer(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                # 112, 112, 32 -> 112, 112, 16
                [1, 16, 1, 1],
                # 112, 112, 16 -> 56, 56, 24
                [6, 24, 2, 2],
                # 56, 56, 24 -> 28, 28, 32
                [6, 32, 3, 2],
                # 28, 28, 32 -> 14, 14, 64
                [6, 64, 4, 2],
                # 14, 14, 64 -> 14, 14, 96
                [6, 96, 3, 1],
                # 14, 14, 96 -> 7, 7, 160
                [6, 160, 3, 2],
                # 7, 7, 160 -> 7, 7, 320
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        # 224, 224, 3 -> 112, 112, 32
        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # 7, 7, 320 -> 7,7,1280
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, 1000),
            nn.Dropout(0.2),
            nn.Linear(1000, 1, bias=False)
        )
        self.linear_1_bias = nn.Parameter(torch.zeros(num_classes-1).float())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        # 1280
        x = x.mean([2, 3])
        x = self.classifier(x)
        x = x + self.linear_1_bias
        return x
    
    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True


def mobilenet_v2(num_classes=101, pretrained=False):
    model = MobileNetV2(num_classes=num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], model_dir='./model_data',
                                              progress=True)
        model.load_state_dict(state_dict)

    # if num_classes!=1000:
    #     model.classifier = nn.Sequential(
    #             nn.Dropout(0.2),
    #             nn.Linear(model.last_channel, num_classes),
    #         )
    return model

def cost_fn(logits, levels):
    val = -torch.sum(F.logsigmoid(logits)*levels + (F.logsigmoid(logits) - logits)*(1-levels), dim=1)
    return torch.mean(val)


def test(model, data_loader, data, logger, mode):
    device = data['DEVICE']
    mae, num_examples = 0, 0
    CA = [0.]*10
    with tqdm(data_loader) as e:
        for inputs in e:
            (feature, age, _) = inputs
            feature = feature.to(device)
            age = age.to(device)

            probas = model(feature)
            probas = torch.sigmoid(probas)
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


def train(model_, dataloaders_, optimizer_, lr_scheduler_, num_epochs_, writers, logger, PATH, data):
    device = data['DEVICE']
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
            logger.info('in {} mode...'.format(phase))

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
                        logits = model_(features)
                        probas = torch.sigmoid(logits)
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

            logger.info('{} Loss: {:.4f} CA_3: {:.4f}, CA_5: {:.4f} MAE:{:.4f}'.format(phase, epoch_loss, CA_3, CA_5, MAE))

            if phase == 'valid':
                val_acc_history.append(MAE.item())
                if MAE < best_mae:
                    best_loss = epoch_loss
                    best_mae, best_epoch, best_CA3, best_CA5 = MAE, epoch, CA_3, CA_5
                    ##### save model #####
                    torch.save(model_.state_dict(), str(PATH.joinpath('best_model.pt')))
                logger.debug('MAE/CA3/CA5: | Current Valid: %.4f/%.4f/%.4f Ep. %d | Best Valid : %.4f/%.4f/%.4f Ep. %d' % (
                    MAE, CA_3, CA_5, epoch+1,
                    best_mae, best_CA3, best_CA5,  best_epoch+1
                ))
        lr_scheduler_.step()
    logger.info('Best val MAE: {:4f}'.format(best_loss))
    return model_, val_acc_history

if __name__ == "__main__":
    model = mobilenet_v2(num_classes=101)
    model = model.cuda()

    x = torch.rand(2, 3, 224, 224)
    # logit, y = model(x)
    # print(y.shape)

    import torchsummary
    torchsummary.summary(model, (3, 224, 224))