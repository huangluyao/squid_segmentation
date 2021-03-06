import torch.nn as nn


def get_loss(cfg, weight=None):

    '''

        :param cfg:
        :param weight: class weighting
        :param ignore_index: class to ignore, 一般为背景id
        :return:
    '''

    assert cfg['loss'] in ['crossentropyloss2D', 'Focal_loss2D']
    if weight is not None:
        assert len(weight) == cfg['n_classes']

        return {
            'crossentropyloss2D': nn.CrossEntropyLoss(weight=weight),}[cfg['loss']]
    else:
        return {
            'crossentropyloss2D': nn.CrossEntropyLoss(), }[cfg['loss']]
