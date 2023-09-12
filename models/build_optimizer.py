# @Time : 2021-12-17 8:32
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : build_optimizer.py

from torch import optim

def build_optimizer(cfg,model):
    # if cfg.Ann:
    #     optimizer_ft=optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
    #     return optimizer_ft

    if cfg.Method=="Baseline":
        ignored_params = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1 * cfg.lr},
            {'params': model.classifier.parameters(), 'lr': cfg.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    if cfg.Method=="OSNet" or cfg.Method=="OSNetTwoView":
        optimizer_ft = optim.SGD(model.parameters(),lr=cfg.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
    elif cfg.Method=="LPN":
        # ignored_params = list(map(id, model.model.fc.parameters() ))
        ignored_params = list()
        for i in range(cfg.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            ignored_params += list(map(id, c.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optim_params = [{'params': base_params, 'lr': 0.1 * cfg.lr}]
        for i in range(cfg.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': cfg.lr})

        optimizer_ft = optim.SGD(optim_params, weight_decay=5e-4, momentum=0.9, nesterov=True)
    elif cfg.Method=="ANN":
        # ignored_params = list(map(id, model.model.fc.parameters() ))
        ignored_params = list()
        for i in range(cfg.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            ignored_params += list(map(id, c.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        optim_params = [{'params': base_params, 'lr': 0.1 * cfg.lr}]
        for i in range(cfg.block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': cfg.lr})

        optimizer_ft = optim.SGD(optim_params, weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        optimizer_ft = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
    # if cfg.optimizer=="adma":
    #     optimizer_ft = optim.Adam(model.parameters(),lr=cfg.lr, weight_decay=5e-4)
    return optimizer_ft