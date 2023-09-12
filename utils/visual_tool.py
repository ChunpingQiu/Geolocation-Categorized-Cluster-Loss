# @Time : 2021-12-17 15:40
# @Author : Wang Zhen
# @Email : wangzhen@stu.xpu.edu.cn
# @File : visual_tool.py
from visualdl import LogWriter
import visualdl

def build_visualDL(logdir='.runs/',name='name'):
    writer = LogWriter(logdir="./log/" + name)
    visualdl.server.app.run(logdir,
                            host="0.0.0.0",
                            port=8080,
                            cache_timeout=20,
                            language=None,
                            public_path=None,
                            api_only=False,
                            open_browser=False)
    return writer
