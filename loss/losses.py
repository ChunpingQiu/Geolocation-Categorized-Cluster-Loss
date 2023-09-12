# @Time : 2021-12-23 8:20
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : losses.py
import torch
from torch import nn
import torch
import torch.nn.functional as F
# import torch.nn as nn
import numpy as np
from config import DefaultConfig,SaveConfig
cfg = DefaultConfig()

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, use_label_smoothing=True):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.to(torch.device('cuda'))
        if use_label_smoothing:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Cluster_loss_TwoView(nn.Module):
    def __init__(self, cfg, dist_type='l2'):
        super(Cluster_loss_TwoView, self).__init__()
        self.margin = cfg.α
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2, label1):
        feat1=Normalize()(feat1)
        feat2=Normalize()(feat2)
        # feat_size = feat1.size()[1]
        # feat_num = feat1.size()[0]
        label_unique=label1.unique()
        feat1_class=[]
        for i in range(len(label_unique)):
            feat1_class.append(feat1[torch.where(label1 == label_unique[i])])

        feat2_class=[]
        for i in range(len(label_unique)):
            feat2_class.append(feat2[torch.where(label1 == label_unique[i])])

        label_num = len(label1.unique())
        # feat1 = feat1.chunk(label_num, 0)
        # feat2 = feat2.chunk(label_num, 0)
        # loss = Variable(.cuda())
        for i in range(label_num):
            center1 = torch.mean(feat1_class[i], dim=0)
            center2 = torch.mean(feat2_class[i], dim=0)
            if self.dist_type == 'l2' or self.dist_type == 'l1':
                if i == 0:
                    dist = max(0, self.dist(center1, center2) - self.margin)
                else:
                    dist += max(0, self.dist(center1, center2) - self.margin)
            elif self.dist_type == 'cos':
                if i == 0:
                    dist = max(0, 1 - self.dist(center1, center2) - self.margin)
                else:
                    dist += max(0, 1 - self.dist(center1, center2) - self.margin)

        return dist
class Cluster_loss(nn.Module):
    def __init__(self, cfg, dist_type='l2'):
        super(Cluster_loss, self).__init__()
        self.margin =  cfg.α
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()
    def distance(self,c1,c2,c3):
        distance=self.dist(c1, c2)+self.dist(c1,c3)+self.dist(c2, c3)
        return distance
    def forward(self, feat1, feat2 , feat3, label1):
        feat1=Normalize()(feat1)
        feat2=Normalize()(feat2)
        feat3 = Normalize()(feat3)
        # feat_size = feat1.size()[1]
        # feat_num = feat1.size()[0]
        label_unique=label1.unique()
        feat1_class=[]
        for i in range(len(label_unique)):
            feat1_class.append(feat1[torch.where(label1 == label_unique[i])])

        feat2_class=[]
        for i in range(len(label_unique)):
            feat2_class.append(feat2[torch.where(label1 == label_unique[i])])

        feat3_class=[]
        for i in range(len(label_unique)):
            feat3_class.append(feat3[torch.where(label1 == label_unique[i])])

        label_num = len(label1.unique())
        # feat1 = feat1.chunk(label_num, 0)
        # feat2 = feat2.chunk(label_num, 0)
        # loss = Variable(.cuda())
        for i in range(label_num):
            center1 = torch.mean(feat1_class[i], dim=0)
            center2 = torch.mean(feat2_class[i], dim=0)
            center3 = torch.mean(feat3_class[i], dim=0)

            if self.dist_type == 'l2' or self.dist_type == 'l1':
                if i == 0:
                    dist = max(0, self.distance(center1, center2,center3)- self.margin)
                else:
                    dist += max(0, self.distance(center1, center2,center3) - self.margin)
            elif self.dist_type == 'cos':
                if i == 0:
                    dist = max(0, 1 - self.distance(center1, center2,center3) - self.margin)
                else:
                    dist += max(0, 1 - self.distance(center1, center2,center3) - self.margin)

        return torch.tensor(dist)


if __name__ == '__main__':


    # entroy = nn.CrossEntropyLoss()
    # Sentroy = CrossEntropyLabelSmooth(num_classes=2,use_gpu=False)
    # input = torch.Tensor([[-0.7715, -0.6205, -0.2562]])
    # target = torch.tensor([0])
    # output1 = entroy(input, target)
    # output2 = Sentroy(input, target)
    # print(output1,output2)  # 采用CrossEntropyLoss计算的结果。
    feat1 = torch.ones((6,512))
    feat2 = torch.ones((6, 512))
    label1=torch.tensor([1,1,2,2,3,3,])
    label2 = torch.tensor([1,1,2,2,3,3,])
    Cl=Cluster_loss()
    out=Cl(feat1, feat2, label1, label2)
    print(out)

