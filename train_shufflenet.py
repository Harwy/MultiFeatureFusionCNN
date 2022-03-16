import sys
import time
import logging
from pathlib import Path
import argparse
import torch
from torch import optim
import json
from tensorboardX import SummaryWriter
from datasets.morph import TrainDataloader, ValidDataloader, TestDataloader
from models.shufflenet_v2 import shufflenetV2 as net
from models.mobilenet import train, test

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Log等级总开关


def parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0005)
    parser.add_argument('--epochs', '-ep', type=int, default=90)
    parser.add_argument('--num_classes', '-n', type=int, default=101)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--numworkers', type=int, default=0)
    parser.add_argument('--data_mode', type=str, required=True, default='rank')
    parser.add_argument('--img_size', '-i', type=int, default=120)
    parser.add_argument('--imp_weight', type=int, default=0)
    parser.add_argument('--outpath', '-o', type=str, required=True)
    parser.add_argument('--description', '-d', type=str, default="resnet34")
    parser.add_argument('--pretrained', '-p', action='store_true')  # 迁移学习或者继续学习
    parser.add_argument('--pretrained_model', '-pm', type=str)
    # shuffleNet 参数
    parser.add_argument('--scale', '-sc', type=float, default=0.5)
    parser.add_argument('--c_tag', '-tag', type=float, default=0.5)
    parser.add_argument('--SE', '-se', action='store_true', help='run with SE module or not') 
    parser.add_argument('--residual', '-re', action='store_true', help='run with residual module or not') 
    args = parser.parse_args()
    return args

def main():
    args = parsers()

    NUM_WORKERS = args.numworkers
    if args.seed == -1:
        RANDOM_SEED = None
    else:
        RANDOM_SEED = args.seed

    IMP_WEIGHT = args.imp_weight
    PATH = Path('Logs', args.outpath)
    if PATH.exists():
        import shutil
        shutil.rmtree(PATH)
    PATH.joinpath('log').mkdir(parents=True, exist_ok=True)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    LOGFILE = str(PATH.joinpath(f'train-{rq}.log'))
    fh = logging.FileHandler(LOGFILE, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(chlr)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(DEVICE)

    writers = SummaryWriter(str(PATH.joinpath('log')))

    logger.info('PyTorch Version: %s' % torch.__version__)
    logger.info('CUDA device available: %s' % torch.cuda.is_available())
    logger.info('Using CUDA device: %s' % DEVICE)
    logger.info('Random Seed: %s' % RANDOM_SEED)
    logger.info('Task Importance Weight: %s' % IMP_WEIGHT)
    logger.info('Output Path: %s' % PATH)
    logger.info('Script: %s' % sys.argv[0])
    logger.info('Description: ' + args.description)

    learning_rate = args.learning_rate
    num_epochs = args.epochs
    num_classes = args.num_classes
    batch_size = args.batch_size
    data_mode = args.data_mode
    img_size = args.img_size
    scale = args.scale
    c_tag = args.c_tag
    SE = args.SE
    residual = args.residual

    train_loader = TrainDataloader(img_size, data_mode, batch_size, num_classes, NUM_WORKERS)
    valid_loader = ValidDataloader(img_size, data_mode, batch_size, num_classes, NUM_WORKERS)
    test_loader = TestDataloader(img_size, data_mode, batch_size, num_classes, NUM_WORKERS)

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

    model = net(scale=scale, c_tag=c_tag, num_classes=num_classes, SE=SE, residual=residual)
    model = model.to(DEVICE)

    if args.pretrained == True:
        model.load_state_dict(torch.load(args.pretrained_model))

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    # 训练
    model, hist = train(model, loader, optimizer, lr_scheduler, num_epochs, writers, logger, PATH, DEVICE)
    json_file = open(str(PATH.joinpath('val_acc_list.json')), mode='w')
    json.dump(hist, json_file, indent=4)
    json_file.close()
    # logger.info("val_acc_history: {}".format(hist))
    # EVALUATE LAST MODEL
    model.eval()
    with torch.set_grad_enabled(False):
        test(model, train_loader, DEVICE, logger, 'train')
        test(model, valid_loader, DEVICE, logger, 'valid')
        test(model, test_loader, DEVICE, logger, 'test')

    # EVALUATE BEST MODEL
    model.load_state_dict(torch.load(str(PATH.joinpath('best_model.pt'))))
    model.eval()
    with torch.set_grad_enabled(False):
        test(model, train_loader, DEVICE, logger, 'train')
        test(model, valid_loader, DEVICE, logger, 'valid')
        test(model, test_loader, DEVICE, logger, 'test')
    
    torch.save(model, str(PATH.joinpath('best_all_model.pth')))
    logger.info('########## 程序结束 ##########')


if __name__ == "__main__":
    main()


