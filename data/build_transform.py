# @Time : 2021-12-17 8:16
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : build_transform.py
from torchvision import transforms

def DefaultTransform(cfg):
    transform_train_list = [

        transforms.Resize((cfg.h, cfg.w), interpolation=3),
        transforms.Pad(cfg.pad, padding_mode='edge'),
        transforms.RandomCrop((cfg.h, cfg.w)),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize([0.4301676, 0.44345695, 0.46232662], [0.21648625, 0.20354147, 0.20370594])
    ]

    transform_satellite_list = [
        transforms.Resize((cfg.h, cfg.w), interpolation=3),
        transforms.Pad(cfg.pad, padding_mode='edge'),
        transforms.RandomAffine(90),
        transforms.RandomCrop((cfg.h, cfg.w)),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize([0.39307848, 0.4174867, 0.4066143], [0.21315724, 0.20504959, 0.19916914])
    ]

    transform_val_list = [
        transforms.Resize(size=(cfg.h, cfg.w), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
        'satellite': transforms.Compose(transform_satellite_list)}

    return data_transforms
def AdvanceTransform(cfg):
    trans = [
            transforms.RandomResizedCrop(cfg.h, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    data_transforms = {
        'train': transforms.Compose(trans),
        'val': transforms.Compose(trans),
        'satellite': transforms.Compose(trans)}
    return data_transforms
def build_train_transform(cfg):
    if cfg.transforms=="default":
        data_transforms=DefaultTransform(cfg)
    elif cfg.transforms=="advance":
        data_transforms = AdvanceTransform(cfg)
    else:
        raise KeyError('Unknown transforms: {}'.format(cfg.transforms))
    return data_transforms

def build_test_transform(cfg,view):
    if cfg.transforms == "default":
        if view == 'satellite':
            data_transforms = transforms.Compose([
                transforms.Resize((cfg.h, cfg.w), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.39307848, 0.4174867, 0.4066143], [0.21315724, 0.20504959, 0.19916914])
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif view == 'drone':
            data_transforms = transforms.Compose([
                transforms.Resize((cfg.h, cfg.w), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.4301676, 0.44345695, 0.46232662], [0.21648625, 0.20354147, 0.20370594])
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    elif cfg.transforms == "advance":
        data_transforms = transforms.Compose([
                transforms.Resize((cfg.h, cfg.w), interpolation=3),
                transforms.ToTensor(),
                # transforms.Normalize([0.4301676, 0.44345695, 0.46232662], [0.21648625, 0.20354147, 0.20370594])
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        raise KeyError('Unknown transforms: {}'.format(cfg.transforms))
    return data_transforms