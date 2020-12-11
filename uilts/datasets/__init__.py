from uilts.datasets.CustomDataset import *


def get_dataset(cfg):
    crop_size = (cfg['image_h'], cfg['image_w'])
    num_class = cfg['n_classes']

    if cfg['dataset']=="Camvid":
        class_dict_path = './database/CamVid/class_dict.csv'

        train_path = "./database/CamVid/train"
        train_label_path = "./database/CamVid/train_labels"
        test_path = "./database/CamVid/test"
        test_label_path = "./database/CamVid/test_labels"

        return CamvidDataset(train_path,train_label_path,cfg["augmentation_path"], class_dict_path, mode="train"), \
               CamvidDataset(test_path, test_label_path, cfg["augmentation_path"], class_dict_path, mode="test")

    if cfg['dataset']=="Squid":

        class_dict_path = './database/Squid/class_dict.csv'

        train_path = "./database/Squid/train"
        train_label_path = "./database/Squid/train_labels"
        test_path = "./database/Squid/test"
        test_label_path = "./database/Squid/test_labels"

        return SquidDataset(train_path,train_label_path,cfg["augmentation_path"], class_dict_path, mode="train"), \
               SquidDataset(test_path, test_label_path, cfg["augmentation_path"], class_dict_path, mode="test")




