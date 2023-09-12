# @Time : 2021-12-18 9:04
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : testmy.py
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import scipy.io
from utils import Auto_Select_GPU
from models import build_model
from data import build_test_transform,build_transform
from config import DefaultConfig,SaveConfig,LoadConfig
from torchvision.datasets import DatasetFolder, ImageFolder
import os

Auto_Select_GPU()
cfg = DefaultConfig()

def get_key (dict, value):
    return int([k for k, v in dict.items() if v == value][0])
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda() # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip
def extract_feature(cfg,model, dataloaders,dataset,view):
    model.eval()
    features = torch.FloatTensor()
    labels = torch.FloatTensor()
    i = 0
    dicxx=dataset.class_to_idx
    for data in dataloaders:
        img, label = data
        img = img.cuda()
        label=torch.tensor([get_key(dicxx,i) for i in label.data])
        # dataloaders.
        out=[]
        for j in range(2):
            if j ==1:
                img=fliplr(img)  #水平反转
            ff = model.featuremaps(img)
            ff = nn.AdaptiveAvgPool2d(1)(ff)
            ff = ff.view(ff.size(0), -1)
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            out.append(ff)

        ff=(out[0]+out[1])/2

        features = torch.cat((features, ff.data.cpu()), 0)
        labels = torch.cat((labels, label.data.cpu()), 0)
        i = i + 1

        print('{}/{}'.format(i, len(dataloaders)))
    return features, labels.numpy().tolist()



if __name__ == "__main__":
    projectname='05221926-osnet_ain_x0_75_geo-Pretrain_2views'
    logdir=os.path.join("/home/qcp/00E/SHS/Light-osnet/output_result",projectname)
    cfg=LoadConfig(os.path.join(logdir,"config.pkl"))
    print(cfg)
    cfg.transforms="default"
    if cfg.Method=="LPN":
        model = build_model(cfg, 701).cuda()
        for i in range(cfg.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            c.classifier = nn.Sequential()
    elif cfg.Method=="ANN":
        model = build_model(cfg, 701).cuda()
    else:

        model = build_model(cfg, 701).cuda()

    model.load_state_dict(torch.load(os.path.join(logdir, 'weights', '130_model.pth')))

    model = model.eval()
    model = model.cuda()

    query_name='drone'
    gallery_name='satellite'
    # query_name = 'satellite'
    # gallery_name = 'drone'

    print("{} -> {}".format(query_name,gallery_name))
    query_path = "/home/qcp/00E/SHS/code/University-1652/University1652-Baseline-master/data/test/query_"+query_name
    query_dataset = ImageFolder(query_path, transform=build_test_transform(cfg,query_name))
    query_dataloaders = DataLoader(query_dataset, batch_size=8, shuffle=False)

    gallery_path = "/home/qcp/00E/SHS/code/University-1652/University1652-Baseline-master/data/test/gallery_"+gallery_name
    gallery_dataset = ImageFolder(gallery_path, transform=build_test_transform(cfg,gallery_name))
    gallery_dataloaders = DataLoader(gallery_dataset, batch_size=8, shuffle=False)

    with torch.no_grad():
        query_feature, query_label = extract_feature(cfg,model, query_dataloaders,query_dataset,view=3)
        gallery_feature, gallery_label = extract_feature(cfg,model, gallery_dataloaders,gallery_dataset,view=1)

    # Save to Matlab for check
    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_path': "gallery_path",
              'query_f': query_feature.numpy(), 'query_label': query_label, 'query_path': "query_path"}
    scipy.io.savemat(os.path.join(logdir,'results',query_name+'-'+gallery_name+'.mat'), result)
    print(projectname)


