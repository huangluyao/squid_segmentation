import os
import json
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from uilts.datasets import get_dataset, denormalize
from uilts.models import get_model
from uilts.log import get_logger
from uilts.evalution import eval_semantic_segmentation
import cv2


def predict(cfg, runid, use_pth='best_train_miou.pth'):

    dataset = cfg['dataset']
    train_logdir = f'run/{dataset}/{runid}'

    test_logdir = os.path.join('./results', dataset, runid)
    logger = get_logger(test_logdir)

    logger.info(f'Conf | use logdir {train_logdir}')
    logger.info(f'Conf | use dataset {cfg["dataset"]}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 测试集
    trainset, testset = get_dataset(cfg)

    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])

    # model
    model = get_model(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(train_logdir, use_pth)))

    # 标签预处理
    pd_label_color = pd.read_csv(trainset.class_dict_path, sep=',')
    name_value = pd_label_color['name'].values
    num_class = len(name_value)
    colormap = []

    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        colormap.append(color)

    cm = np.array(colormap).astype('uint8')

    test_miou = 0

    for i, (valImg, valLabel) in enumerate(test_loader):
        valImg = valImg.to(device)
        valLabel = valLabel.long().to(device)
        out = model(valImg)
        out = F.log_softmax(out, dim=1)

        pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
        pre = cm[pre_label]

        src = denormalize(valImg.cpu(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        src = np.transpose(src.squeeze().data.numpy(), [1, 2, 0])

        pre_label = np.expand_dims(pre_label,axis=-1)
        result =pre_label*src + (1-pre_label)*(src * 0.3 +pre * 0.7)
        result = result.astype(np.uint8)
        cv2.imwrite(test_logdir + '/' + str(i) + '.png', result)
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = valLabel.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrix = eval_semantic_segmentation(pre_label, true_label, cfg["n_classes"])
        test_miou = eval_metrix['miou'] + test_miou

    logger.info(f'Test | Test Mean IU={test_miou / (len(test_loader)):.5f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument("-id", type=str, help="predict id")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/Squid_UNet.json",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    args.id = '2020-12-11-19-05-3988'

    predict(cfg, args.id)
