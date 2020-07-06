import os
import time

### 模型、日志保存路径
model_save_path = './weights/last.pt'
log_folder = 'log'
log_file_path = os.path.join(log_folder , 'log_{}.txt'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime())))
###

ALREADY_SHOWED_SAMPLE = False
DATA_PATH = '/home/user/dataset/voc2007'