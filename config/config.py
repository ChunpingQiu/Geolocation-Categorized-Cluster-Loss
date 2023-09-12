# @Time : 2021-12-17 8:06
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : config.py
# coding=utf-8

import pickle
import os
import datetime
class DefaultConfig:
    """ Base configuration class for perparameter settings.
    All new configuration should be derived from this class.
    """

    def __init__(self):
        # 获得当前时间
        now = datetime.datetime.now()
        # 转换为指定的格式:
        TimeStamp = now.strftime("%m%d%H%M")
        self.Method = "osnet_ain_x0_75_geo"
        self.PROJECT_NAME = TimeStamp+'-'+self.Method+'-wh1_Pretrain'  # project name
        self.epochs=131
        self.views=2
        self.lr=0.001
        self.droprate=0.75
        self.pretrained=True
        self.fp16 = False
        self.transforms='default'
        self.criterion="CrossEntropy"
        self.loss = "cluster"
        self.λ=1
        self.α = 0.1
        self.pairs = 2
        self.pool='avg'
        self.data_dir = '/home/qcp/00E/SHS/code/University-1652/University1652-Baseline-master/data/train'
        self.test_data_dir = '/home/qcp/00E/SHS/code/University-1652/University1652-Baseline-master/data/test'
        self.categories=8
        self.stride=1
        self.pad=10
        self.h=256
        self.w=256
        self.erasing_p=0
        self.use_dense=False
        self.warm_epoch=0
        self.resume=False
        self.share=True
        self.extra_Google=True


    def __str__(self):
        for item in self.__dict__.items():
            print("%s:%s" % item)
        return "------------"

def SaveConfig(config,path):
    print("Save Config:",path)
    pkl_dir=os.path.join(path, "config.pkl")
    output_hal = open(pkl_dir, 'wb')
    str = pickle.dumps(config)
    output_hal.write(str)
    output_hal.close()
    txt_dir=os.path.join(path, "config.txt")
    file = open(txt_dir, 'w')
    # 遍历字典的元素，将每项元素的key和value分拆组成字符串，注意添加分隔符和换行符
    for item in config.__dict__.items():
        file.write('%s : %s \n' % item)
    file.close()
def LoadConfig(path):
    with open(path, 'rb') as file:
        self = pickle.loads(file.read())
    print()
    return self
if __name__ == '__main__':
    pass