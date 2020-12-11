from .BiSeNet import BiSeNet
from .DFANet import dfaNet
from .UNet_Res import UNet

def get_model(cfg):
    return {'UNet': UNet,
            'BiseNet': BiSeNet,
            'dfaNet': dfaNet,
            }[cfg["model_name"]](num_classes=cfg["n_classes"])