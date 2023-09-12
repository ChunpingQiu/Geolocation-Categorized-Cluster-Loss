# @Time : 2021-12-23 8:27
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : build_loss.py
from .losses import CrossEntropyLabelSmooth,Cluster_loss,Cluster_loss_TwoView
from torch import nn
from config import DefaultConfig,SaveConfig
cfg = DefaultConfig()

def build_loss(cfg,class_num):

    if cfg.views==2:
        loss_intra=Cluster_loss_TwoView(cfg)
    elif cfg.views==3:
        loss_intra=Cluster_loss(cfg)
    else:
        raise Exception("Not support views")

    if cfg.criterion=="CrossEntropy":
        loss_inter = nn.CrossEntropyLoss()
    elif cfg.criterion=="CrossEntropyLabelSmooth":
        loss_inter = CrossEntropyLabelSmooth(num_classes=class_num)
    else:
        raise Exception("No find useful criterion function")

    return [loss_inter,loss_intra]