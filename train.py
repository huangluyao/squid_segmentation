import argparse, json, time, random, os
import shutil
from uilts.log import get_logger
from uilts.datasets import get_dataset
from uilts.models import get_model
from uilts.loss import get_loss
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from uilts.evalution import *


def run(cfg, logger):
    # 1. The dataset name used
    logger.info(f'Conf | use dataset {cfg["dataset"]}')
    logger.info(f'Conf | use batch_size {cfg["batch_size"]}')
    logger.info(f'Conf | use model_name {cfg["model_name"]}')

    # 2. load dataset
    trainset, valset = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    # 3. load_model
    model = get_model(cfg)

    # 4. Whether to use multi-gpu training
    gpu_ids = [int(i) for i in list(cfg['gpu_ids'])]
    logger.info(f'Conf | use GPU {gpu_ids}')
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(cfg["device"])

    # 5. optimizer and learning rate decay
    logger.info(f'Conf | use optimizer Adam, lr={cfg["lr"]}, weight_decay={cfg["weight_decay"]}')
    logger.info(f'Conf | use step_lr_scheduler every {cfg["lr_decay_steps"]} steps decay {cfg["lr_decay_gamma"]}')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    # 6. loss function and class weight balance
    logger.info(f'Conf | use loss function {cfg["loss"]}')
    criterion = get_loss(cfg).to(cfg['device'])

    # 7. train and val
    logger.info(f'Conf | use epoch {cfg["epoch"]}')
    best = 0.
    for epoch in range(cfg['epoch']):
        model.train()
        train_loss = 0
        train_miou = 0

        nLen = len(train_loader)
        batch_bar = tqdm(enumerate(train_loader), total=nLen)
        for i, (img_data, img_label) in batch_bar:
            # load data to gpu
            img_data = img_data.to(cfg['device'])
            img_label = img_label.to(cfg['device'])
            # forward
            out = model(img_data)
            # calculate loss
            loss = criterion(out, img_label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # evaluate
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label, cfg["n_classes"])
            train_miou += eval_metrix['miou']
            batch_bar.set_description('|batch[{}/{}]|train_loss {: .8f}|'.format(i + 1, nLen, loss.item()))

        logger.info(f'Iter | [{epoch + 1:3d}/{cfg["epoch"]}] train loss={train_loss / len(train_loader):.5f}')
        logger.info(f'Test | [{epoch + 1:3d}/{cfg["epoch"]}] Train Mean IU={train_miou / len(train_loader):.5f}')

        miou = train_miou / len(train_loader)
        if best <= miou:
            best = miou
            torch.save(model.state_dict(), os.path.join(cfg['logdir'], 'best_train_miou.pth'))

        net = model.eval()
        eval_loss = 0
        eval_miou = 0

        for j, (valImg, valLabel) in enumerate(val_loader):
            valImg = valImg.to(cfg['device'])
            valLabel = valLabel.to(cfg['device'])

            out = net(valImg)
            loss = criterion(out, valLabel)
            eval_loss = loss.item() + eval_loss
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = valLabel.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrics = eval_semantic_segmentation(pre_label, true_label, cfg["n_classes"])
            eval_miou = eval_metrics['miou'] + eval_miou

        logger.info(f'Iter | [{epoch + 1:3d}/{cfg["epoch"]}] valid loss={eval_loss / len(val_loader):.5f}')
        logger.info(f'Test | [{epoch + 1:3d}/{cfg["epoch"]}] Valid Mean IU={eval_miou / len(val_loader):.5f}')


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config",
                        nargs="?",
                        type=str,
                        default="configs/Squid_UNet.json",
                        help="Configuration to use")

    args = parser.parse_args()

    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    # Training Record
    logdir = f'run/{cfg["dataset"]}/{time.strftime("%Y-%m-%d-%H-%M")}-{random.randint(1000,10000)}'
    os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)

    logger.info(f'Conf | use logdir {logdir}')
    cfg['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg['logdir'] = logdir

    run(cfg, logger)




