# @Time : 2022-01-05 19:15
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : make_dir.py
# @Project : ANN-MetricLearning
import os

def make_dir(projectname=None,base='/storage4/wangzhen/code/DSCAN/logs'):
    logdir=os.path.join(base,projectname)
    folder = os.path.exists(logdir)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(logdir)  # makedirs 创建文件时如果路径不存在会创建这个路径
    weight_dir=os.path.join(logdir,"weights")
    folder = os.path.exists(weight_dir)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(weight_dir)  # makedirs 创建文件时如果路径不存在会创建这个路径
    resluts_dir=os.path.join(logdir,"results")
    folder = os.path.exists(resluts_dir)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(resluts_dir)  # makedirs 创建文件时如果路径不存在会创建这个路径
    return logdir,weight_dir,resluts_dir
