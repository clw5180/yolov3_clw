import os
import time
import torch

### pytorch version
import re
match = re.match(r"(\d\.\d\.\d)(.*)", torch.__version__)
pytorch_version_major, pytorch_version_minor, patch = map(int, match.group(1).split("."))
###

### 模型、日志保存路径
model_save_path = './weights/last.pt'
log_folder = 'log'
log_file_path = os.path.join(log_folder , 'log_{}.txt'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime())))
###

ALREADY_SHOWED_SAMPLE = False
DATA_PATH = '/home/user/dataset/voc2007'