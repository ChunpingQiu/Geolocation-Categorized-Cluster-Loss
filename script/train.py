# @Time : 2021-12-17 8:37
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : train.py
import time
import torch
from torch import nn
from utils import meter
from utils.meter import AverageMeter
from torch.cuda.amp import autocast as autocast, GradScaler
from prettytable import PrettyTable

def train_Metric_twoview(cfg,model,dataloaders, loss, optimizer):
    model.train()
    loss_meter = AverageMeter()
    loss_meter_h = AverageMeter()
    loss_meter_std = AverageMeter()
    acc_satellite_meter = AverageMeter()
    acc_drone_meter = AverageMeter()
    for batch_idx, (s, d, st, id) in enumerate(dataloaders):

        s = torch.cat(s, dim=0).cuda()
        d = torch.cat(d, dim=0).cuda()
        id = torch.cat(id, dim=0).cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward

        outputs1, outputs2,feature1,feature2 = model(s,d)
        _, preds1 = torch.max(outputs1.data, 1)
        _, preds2 = torch.max(outputs2.data, 1)

        loss1 = loss[0](outputs1, id)
        loss2 = loss[0](outputs2, id)

        loss_intra = loss[1](feature1, feature2, id)

        all_loss = loss1 + loss2 + cfg.λ * loss_intra

        loss0 = torch.stack([loss1 ,loss2])
        loss_std=torch.std(loss0)

        all_loss.backward()
        optimizer.step()

        # statistics
        loss_meter.update(all_loss.item(), s.shape[0])
        loss_meter_h.update(loss_intra.item(), s.shape[0])
        loss_meter_std.update(loss_std.item(), s.shape[0])
        acc_satellite_meter.update((preds1 == id.data).float().cpu().mean().numpy(), 1)
        acc_drone_meter.update((preds2 == id.data).float().cpu().mean().numpy(), 1)

    return {"Train/loss":loss_meter.avg,
            "Train/loss_intra":loss_meter_h.avg,
            "Train/loss_std": loss_meter_std.avg,
            "Acc/Satellite": acc_satellite_meter.avg,
            "Acc/Drone":acc_drone_meter.avg,
            }

def train_Metric(cfg,model,dataloaders, loss, optimizer,scaler):
    model.train()
    loss_meter = AverageMeter()
    loss_meter_h = AverageMeter()
    loss_meter_std = AverageMeter()
    acc_satellite_meter = AverageMeter()
    acc_drone_meter = AverageMeter()
    acc_street_meter = AverageMeter()
    acc_google_meter = AverageMeter()
    for batch_idx, (s, d, st, id) in enumerate(dataloaders):
        # print(batch_idx)
        s = torch.cat(s, dim=0).cuda()
        d = torch.cat(d, dim=0).cuda()
        st = torch.cat(st, dim=0).cuda()
        id = torch.cat(id, dim=0).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        with autocast(enabled=cfg.fp16):
            outputs1, outputs2,outputs3,feature1,feature2,feature3 = model(s,d,st)

            _, preds1 = torch.max(outputs1.data, 1)
            _, preds2 = torch.max(outputs2.data, 1)
            _, preds3 = torch.max(outputs3.data, 1)

            loss1 = loss[0](outputs1, id)
            loss2 = loss[0](outputs2, id)
            loss3 = loss[0](outputs3, id)

            loss_intra=loss[1](feature1,feature2,feature3,id)


        all_loss =  loss1 + loss2 + loss3 +cfg.λ * loss_intra

        loss0 = torch.stack([loss1 ,loss2,loss3])
        loss_mean=loss_intra
        loss_std=torch.std(loss0)

        scaler.scale(all_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # statistics
        loss_meter.update(all_loss.item(), s.shape[0])
        loss_meter_h.update(loss_mean.item(), s.shape[0])
        loss_meter_std.update(loss_std.item(), s.shape[0])
        acc_satellite_meter.update((preds1 == id.data).float().cpu().mean().numpy(), 1)
        acc_drone_meter.update((preds2 == id.data).float().cpu().mean().numpy(), 1)
        acc_street_meter.update((preds3 == id.data).float().cpu().mean().numpy(), 1)
    return {"Train/loss":loss_meter.avg,
            "Train/loss_intra":loss_meter_h.avg,
            "Train/loss_std": loss_meter_std.avg,
            "Acc/Satellite": acc_satellite_meter.avg,
            "Acc/Drone":acc_drone_meter.avg,
            "Acc/Street": acc_street_meter.avg,}