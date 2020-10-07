# yolov3_clw
yolov3论文复现


### 训练相关
1. cd tools，然后python xml2txt.py
2.


### 测试相关
目前测试只支持resize，否则对于平移这种，还需要在transform.py中传回用于后续bbox坐标变换用的ratio, pad等，比较麻烦；

### 日志相关 
日志格式：log_yyyymmdd_mmss.txt
保留的日志格式：yyyymmdd_xxxxxxx
批量删除：rm log_*