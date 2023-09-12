# @Time : 2022-03-16 9:36
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : configYML.py
# @Project : Light-osnet
import yaml
f = open(r'config.yml')  #  传入文件路径
y = yaml.load(f,Loader=yaml.FullLoader)
print (y)
print(y.Method)