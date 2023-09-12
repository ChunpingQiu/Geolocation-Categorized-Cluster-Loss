# @Time : 2021-12-17 8:06
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : main.py
from config import DefaultConfig,SaveConfig
import torch
from torch.optim import lr_scheduler
from utils import Auto_Select_GPU
from script import train_Metric,train_Metric_twoview
from data import BuildTrainDatasetTwoView
from models import build_optimizer
from models import build_model
from loss import build_loss
from torch import nn
import os,shutil
from torch.utils.tensorboard import SummaryWriter
from utils import make_dir

Auto_Select_GPU()

cfg = DefaultConfig()

UnversityDataset=BuildTrainDatasetTwoView(cfg)
dataloaders=UnversityDataset.dataloader()
class_num=UnversityDataset.class_num

model=build_model(cfg,class_num).cuda()

optimizer=build_optimizer(cfg,model)
loss=build_loss(cfg,class_num)

scheduler=lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1,last_epoch=-1)


if __name__ == '__main__':
    # torch.backends.cudnn.benchmark = True

    writer=SummaryWriter(comment=cfg.PROJECT_NAME)
    basedir='/home/qcp/00E/SHS/Geolocation_Categorized_Cluster_Loss/output_result'
    logdir,weight_dir,resluts_dir=make_dir(cfg.PROJECT_NAME,basedir)
    print(logdir)
    SaveConfig(cfg, logdir)
    shutil.copy("./config/config.py", os.path.join(logdir,'config.py'))
    for epoch in range(cfg.epochs):
        info = train_Metric_twoview(cfg, model, dataloaders, loss, optimizer)
        scheduler.step()
        writer.add_scalar("Train/LR", scheduler.get_lr()[0], epoch)
        print(epoch,info)
        for (key,value) in info.items():
            writer.add_scalar(key, value, epoch)
        if epoch%5==0:
            torch.save(model.state_dict(), os.path.join(weight_dir,str(epoch)+'_model.pth'))
    writer.add_hparams(cfg.__dict__, info)