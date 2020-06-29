import os
import time

### 模型、日志保存路径
last_model_path = './weights/last.pt'
log_folder = 'log'
log_file_path = os.path.join(log_folder , 'log_{}.txt'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime())))
###
