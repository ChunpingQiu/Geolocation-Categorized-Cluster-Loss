# @Time : 2021-12-17 8:16
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : make_dataloader.py
from .folder import ImageFolder
from .build_transform import build_train_transform,build_test_transform
import os
import torch
from .university import UniversityTwoViewDataset


class BuildTrainDatasetTwoView:
    def __init__(self,cfg):
        self.cfg=cfg
        data_transforms=build_train_transform(self.cfg)
        self.image_datasets = {}
        data_dir=self.cfg.data_dir
        self.dataset=UniversityTwoViewDataset(data_dir,cfg.pairs,data_transforms)
        self.class_num = len(self.dataset)

    def dataloader(self):
        dataloaders = torch.utils.data.DataLoader(self.dataset, batch_size=self.cfg.categories,
                                                      shuffle=True, num_workers=4, pin_memory=False)

        return dataloaders

class BuildTrainDataset:
    def __init__(self,cfg):
        self.cfg=cfg
        data_transforms=build_train_transform(self.cfg)
        self.image_datasets = {}
        data_dir=self.cfg.data_dir
        self.image_datasets['satellite'] = ImageFolder(os.path.join(data_dir, 'satellite'),
                                                  data_transforms['satellite'])
        self.image_datasets['street'] = ImageFolder(os.path.join(data_dir, 'street'),
                                               data_transforms['train'])
        self.image_datasets['drone'] = ImageFolder(os.path.join(data_dir, 'drone'),
                                              data_transforms['train'])
        self.image_datasets['google'] = ImageFolder(os.path.join(data_dir, 'google'),
                                               data_transforms['train'])
        pass
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['satellite', 'street', 'drone', 'google']}
        self.class_num = len(self.image_datasets['drone'].classes)

    def dataloader(self):
        dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=self.cfg.categories,
                                                      shuffle=True, num_workers=4, pin_memory=False)
                       # 8 workers may work faster
                       for x in ['satellite', 'street', 'drone', 'google']}

        return dataloaders


class BuildTestDataset:
    def __init__(self,cfg):
        self.cfg=cfg
        data_transforms=build_test_transform(self.cfg)
        self.image_datasets = {}
        data_dir=self.cfg.test_data_dir
        self.image_datasets = {x: ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                          ['gallery_satellite', 'gallery_drone', 'gallery_street', 'query_satellite', 'query_drone',
                           'query_street']}
    def dataloader(self):

        dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=self.cfg.categories,
                                                      shuffle=False, num_workers=16) for x in
                       ['gallery_satellite', 'gallery_drone', 'gallery_street', 'query_satellite', 'query_drone',
                        'query_street']}

        return dataloaders
