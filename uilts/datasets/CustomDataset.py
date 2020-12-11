import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2 as cv
from uilts.parse_cfg import parse_json
import configs as cfg
from augmentation.pipelines.compose import Compose
import pandas as pd


class LabelProcessor:
    '''标签预处理'''
    def __init__(self, file_path):
        colormap = self.read_color_map(file_path)
        # 对标签做编码，返回哈希表
        self.cm2lbl = self.encode_label_pix(colormap)

    # 将mask中的RGB转成编码的label
    def encode_label_img(self, img):
        data = np.array(img, np.int32)
        idx = (data[:, :, 0] * 256+data[:, :, 1])*256 + data[:, :, 2]
        # 返回编码后的label
        return np.array(self.cm2lbl[idx], np.int64)

    # 返回一个哈希映射  再 3维256 空间中
    @staticmethod
    def encode_label_pix(colormap):
        cm2lbl = np.zeros(256**3)  # 3维的256的空间 打成一维度
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i
        return cm2lbl

    # 读取csv文件
    @staticmethod
    def read_color_map(file_path):
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []

        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)

        return colormap


class CamvidDataset(Dataset):

    def __init__(self, img_path, label_path, json_path, class_dict_path, mode="train"):
        self.imgs = self.read_file(img_path)
        self.labels = self.read_file(label_path)

        assert len(self.imgs) == len(self.labels), "label 和 image 数据长度不同"

        config = parse_json(json_path)
        if mode == 'train':
            self.train_pipeline = Compose(config['train'])
        else:
            self.train_pipeline = Compose(config['test'])

        self.tf = transforms.Compose([
            lambda x:torch.tensor(x, dtype=torch.float32)])

        self.class_dict_path = class_dict_path
        self.label_processor = LabelProcessor(class_dict_path)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img = self.imgs[item]
        label = self.labels[item]

        image = cv.imread(img)
        label = cv.imread(label)[..., ::-1]   # BGR 2 RGB

        img, label = self.img_transform(image, label)

        return img, label

    def read_file(self, path):
        '''从文件夹中读取数据'''
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, file) for file in files_list]
        file_path_list.sort()
        return file_path_list

    def img_transform(self, image, mask):
        '''图像数据预处理并转成tensor格式'''
        # 获取图像信息
        data = {"type": "segmentation"}
        data["image"] = image
        data["mask"] = mask

        # 数据增强
        augment_result = self.train_pipeline(data)

        image = augment_result["image"]
        mask = augment_result["mask"]

        # 转成tensor格式
        image = self.tf(np.transpose(image, (2, 0, 1)))

        # 对标签进行编码，转成tensor
        mask = self.label_processor.encode_label_img(mask)
        mask = torch.from_numpy(mask)

        return image, mask


class VOCDataset(Dataset):
    def __init__(self, voc_path, json_path, mode="train"):
        self.voc_path = voc_path
        file_path = os.path.join(voc_path, 'ImageSets/Segmentation')

        self.imgs, self.labels = self.read_file(file_path, mode)
        assert len(self.imgs) == len(self.labels), "label 和 image 数据长度不同"

        config = parse_json(json_path)
        if mode == 'train':
            self.train_pipeline = Compose(config['train'])
        else:
            self.train_pipeline = Compose(config['test'])

        self.tf = transforms.Compose([
            lambda x:torch.tensor(x, dtype=torch.float32)])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        image = self.imgs[item]
        label = self.labels[item]

        image = cv.imread(image)
        label = cv.imread(label)[..., ::-1]

        img, label = self.img_transform(image, label)

        return img, label

    def img_transform(self, image, mask):
        '''图像数据预处理并转成tensor格式'''
        # 获取图像信息
        data = {"type": "segmentation"}
        data["image"] = image
        data["mask"] = mask

        # 数据增强
        augment_result = self.train_pipeline(data)

        image = augment_result["image"]
        mask = augment_result["mask"]

        # 转成tensor格式
        image = self.tf(np.transpose(image, (2, 0, 1)))

        # 对标签进行编码，转成tensor
        mask = label_processor.encode_label_img(mask)
        mask = torch.from_numpy(mask)

        return image, mask

    def read_file(self, file_path, mode):
        if mode == "train":
            imgs_path = os.path.join(file_path, 'train.txt')
        else:
            imgs_path =os.path.join(file_path, 'val.txt')

        f_imgs = open(imgs_path, 'r')

        img_names = [img[:-1] for img in f_imgs.readlines()]
        f_imgs.close()

        imgs = [os.path.join(self.voc_path, 'JPEGImages/%s.jpg' % (img)) for img in img_names]
        labels = [os.path.join(self.voc_path, 'SegmentationClass/%s.png' % (img)) for img in img_names]
        return imgs, labels


class SquidDataset(Dataset):
    def __init__(self, img_path, label_path, json_path, class_dict_path, mode="train"):
        self.imgs = self.read_file(img_path)
        self.labels = self.read_file(label_path)

        assert len(self.imgs) == len(self.labels), "label 和 image 数据长度不同"

        config = parse_json(json_path)
        if mode == 'train':
            self.train_pipeline = Compose(config['train'])
        else:
            self.train_pipeline = Compose(config['test'])

        self.tf = transforms.Compose([
            lambda x: torch.tensor(x, dtype=torch.float32)])

        self.class_dict_path = class_dict_path
        self.label_processor = LabelProcessor(class_dict_path)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img = self.imgs[item]
        label = self.labels[item]

        image = cv.imread(img)
        label = cv.imread(label)[..., ::-1]  # BGR 2 RGB

        img, label = self.img_transform(image, label)

        return img, label

    def read_file(self, path):
        '''从文件夹中读取数据'''
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, file) for file in files_list]
        file_path_list.sort()
        return file_path_list

    def img_transform(self, image, mask):
        '''图像数据预处理并转成tensor格式'''
        # 获取图像信息
        data = {"type": "segmentation"}
        data["image"] = image
        data["mask"] = mask

        # 数据增强
        augment_result = self.train_pipeline(data)

        image = augment_result["image"]
        mask = augment_result["mask"]

        # 转成tensor格式
        image = self.tf(np.transpose(image, (2, 0, 1)))

        # 对标签进行编码，转成tensor
        mask = self.label_processor.encode_label_img(mask)
        mask = torch.from_numpy(mask)

        return image, mask


def denormalize(x_hat, mean=[0.2826372, 0.2826372, 0.2826372], std=[0.30690703, 0.30690703, 0.30690703]):

    mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    x = x_hat * std + mean
    return x*255


def linknet_class_weight(num_classes):
    p_class = num_classes / num_classes.sum()
    return 1 / (np.log(1.02 + p_class))


def compute_weight(root, n_classes):
    num_classes = np.zeros(n_classes)
    for image in os.listdir(root):
        image = Image.open(os.path.join(root, image))
        image = np.asarray(image)   # 360, 480
        image = np.asarray(image).reshape(-1)   # 360 * 480
        num = np.bincount(image)        # len = 12
        num_classes += num      # 每个类别出现的总次数

    weight = linknet_class_weight(num_classes)

    return torch.Tensor(weight.tolist())


if __name__ == "__main__":
    """验证Camvid数据集"""
    # test = CamvidDataset(cfg.train_path, cfg.train_label_path, cfg.json_path, mode="train")


    """验证VOC数据集"""
    # test = VOCDataset(cfg.voc_path, cfg.json_path)
    # from torch.utils.data import DataLoader

    """验证鱿鱼数据集"""
    class_dict_path = '../../database/Squid/class_dict.csv'

    train_path = "../../database/Squid/train"
    train_label_path = "../../database/Squid/train_labels"
    test_path = "../../database/Squid/test"
    test_label_path = "../../database/Squid/test_labels"

    augmentation_path = "../../configs/imagenet.json"
    test = SquidDataset(train_path, train_label_path, augmentation_path, class_dict_path, mode="train")

    test_db = DataLoader(test, batch_size=1)

    label_processor = LabelProcessor(class_dict_path)

    cm = np.array(label_processor.read_color_map(class_dict_path))

    for img, label in test_db:
        images = denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images = images.numpy()
        labels = label.numpy()
        for image, label in zip(images, labels):
            image = np.transpose(image, (1, 2, 0))
            label = label.astype(np.int32)
            label = cm[label][..., ::-1]

            cv.imshow("img", image.astype(np.uint8))
            cv.imshow('lable', label.astype(np.uint8))
            cv.waitKey()


