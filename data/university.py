# @Time : 2021-12-24 8:51
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : university.py
# @Project : ANN-MetricLearning
from torch.utils.data import Dataset
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import os
import glob
import os.path as osp
from PIL import Image
import torch
from .build_transform import build_train_transform
from config import DefaultConfig,SaveConfig
cfg = DefaultConfig()
class UniversityTwoViewDataset(Dataset):#需要继承torch.utils.data.Dataset
    def __init__(self, basepath,N=cfg.pairs, transform=ToTensor()):
        #load drone view images
        view = ['drone', 'satellite', 'google', 'street']
        self.drone_dir=os.path.join(basepath,view[0])
        self.drone_img_paths = np.array(glob.glob(osp.join(self.drone_dir, '*/*')))
        self.drone_IDs=[os.path.normpath(i).split(os.sep)[-2] for i in self.drone_img_paths]
        self.drone_IDs=np.array(self.drone_IDs)
        #load satellite view images
        self.satellite_dir=os.path.join(basepath,view[1])
        self.satellite_img_paths = np.array(glob.glob(osp.join(self.satellite_dir, '*/*')))
        self.satellite_IDs=[os.path.normpath(i).split(os.sep)[-2] for i in self.satellite_img_paths]
        self.satellite_IDs=np.array(self.satellite_IDs)
        #load google view images
        self.google_dir=os.path.join(basepath,view[2])
        self.google_img_paths = np.array(glob.glob(osp.join(self.google_dir, '*/*')))
        self.google_IDs=[os.path.normpath(i).split(os.sep)[-2] for i in self.google_img_paths]
        self.google_IDs=np.array(self.google_IDs)
        #load street view images
        self.street_dir=os.path.join(basepath,view[3])
        self.street_img_paths = np.array(glob.glob(osp.join(self.street_dir, '*/*')))
        self.street_IDs=[os.path.normpath(i).split(os.sep)[-2] for i in self.street_img_paths]
        self.street_IDs=np.array(self.street_IDs)
        #Samplar
        self.N = N #each person select N images
        #build transform
        self.index2class=np.unique(self.drone_IDs)
        self.transform_drone = transform["train"] #变换
        self.transform_satellite = transform["satellite"]  # 变换
        pass
    def __getitem__(self, index):

        select_ID=self.index2class[index]
        select_sallite=[]
        select_drone = []
        select_street=[]
        select_google=[]
        select_index=[]
        for i in range(self.N):
            #satellite
            select_s_pool=self.satellite_img_paths[np.where(self.satellite_IDs==select_ID)]
            select_s=np.random.choice(select_s_pool,size=1)[0]
            s_img = Image.open(select_s).convert('RGB')
            select_sallite.append(self.transform_satellite(s_img))
            #drone
            select_d_pool = self.drone_img_paths[np.where(self.drone_IDs == select_ID)]
            select_d=np.random.choice(select_d_pool,size=1)[0]
            d_img = Image.open(select_d).convert('RGB')
            select_drone.append(self.transform_drone(d_img))
            #street
            select_st_pool = self.street_img_paths[np.where(self.street_IDs == select_ID)]
            select_st=np.random.choice(select_st_pool,size=1)[0]
            st_img = Image.open(select_st).convert('RGB')
            select_street.append(self.transform_drone(st_img))
            #google
            # select_g_pool = self.drone_img_paths[np.where(self.google_IDs == select_ID)]
            # select_g=np.random.choice(select_g_pool,size=1)[0]
            # g_img = Image.open(select_g).convert('RGB')
            # select_google.append(self.transform_drone(g_img))
            select_index.append(index)
        return select_sallite,select_drone,select_street,select_index
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(np.unique(self.drone_IDs))
if __name__ == '__main__':
    basepath='/storage4/wangzhen/University-Release/train'
    dataset=UniversityTwoViewDataset(basepath)
    dataloader=DataLoader(dataset,batch_size=4,shuffle=True)
    for batch_idx, (s, d,id) in enumerate(dataloader):
        s=torch.cat(s,dim=0).cuda()
        d=torch.cat(d,dim=0).cuda()
        id=torch.cat(id,dim=0).cuda()

        a=1
        s=s.cuda()
        d=d.cuda()
        id=id.cuda()
        a=1
