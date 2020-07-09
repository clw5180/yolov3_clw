from utils.voc_eval import voc_eval
from utils.utils import select_device
from model.models import Darknet
from dataset.datasets import VocDataset
from utils.utils import non_max_suppression, load_classes, ap_per_class, xywh2xyxy, bbox_iou, write_to_file, scale_coords
from utils.parse_config import parse_data_cfg
from utils.globals import DATA_PATH


import shutil
import argparse
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import time
import numpy as np
import torch.nn as nn
import cv2

result_path = './result'
classes_pred = set()
if os.path.exists(result_path):
    shutil.rmtree(result_path)
os.makedirs(result_path)

def test(cfg,
         data,
         batch_size,
         img_size,
         conf_thres,
         iou_thres,
         nms_thres,
         src_txt_path='./valid.txt',
         dst_path='./output',
         weights=None,
         model=None,
         log_file_path='log.txt'):

    # 0、初始化一些参数
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    class_names = load_classes(data['names'])


    # 1、加载网络
    if model is None:
        device = select_device(opt.device)
        model = Darknet(cfg)
        if weights.endswith('.pt'):      # TODO: .weights权重格式
            model.load_state_dict(torch.load(weights)['model'])  # TODO：map_location=device ？
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)  # clw note: 多卡
    else:
        device = next(model.parameters()).device  # get model device
    model.to(device).eval()

    # 2、加载数据集
    test_dataset = VocDataset(src_txt_path, img_size, with_label=True, is_training=False)
    dataloader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8,    # TODO
                            collate_fn=test_dataset.test_collate_fn,   # TODO
                            pin_memory=True)

    # 3、预测，前向传播
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@{}'.format(iou_thres), 'F1')


    pbar = tqdm(dataloader)
    for i, (img_tensor, _, img_path, shapes) in enumerate(pbar):
        start = time.time()
        img_tensor = img_tensor.to(device)   # (bs, 3, 416, 416)

        # Disable gradients
        with torch.no_grad():
            # (1) Run model
            output = model(img_tensor)[0]
            # (2) NMS
            nms_output = non_max_suppression(output, conf_thres, nms_thres)  # list (64,)
            s = 'time use per batch: %.3fs' % (time.time() - start)

        pbar.set_description(s)

        for batch_idx, pred in enumerate(nms_output):  # pred: (bs, 7) -> xyxy, obj_conf*class_conf, class_conf, cls_idx
            ################################################
            if pred is None:
                continue
            bboxes_prd = torch.cat((pred[:, 0:5], pred[:, 6].unsqueeze(1)), dim=1).cpu().numpy()

            ###### clw note: coord transform to origin size(because of resize and so on....) is really important !!!
            scale_coords(img_tensor[batch_idx].shape[1:], bboxes_prd, shapes[batch_idx][0], shapes[batch_idx][1])  # to original shape
            ######

            for bbox in bboxes_prd:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])

                class_name = class_names[class_ind]
                classes_pred.add(class_name)
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                s = ' '.join([str(img_path[batch_idx]), str(score), xmin, ymin, xmax, ymax]) + '\n'

                with open(os.path.join(result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(s)
            ################################################
    return calc_APs()


def calc_APs(iou_thresh=0.5, use_07_metric=True):
    """
    计算每个类别的ap值
    :param iou_thresh:
    :param use_07_metric:
    :return:dict{cls:ap}
    """
    filename = os.path.join(result_path, 'comp4_det_test_{:s}.txt')
    cachedir = os.path.join(result_path, 'cache')
    annopath = os.path.join(DATA_PATH, 'val', '{:s}.xml')
    imagesetfile = os.path.join(os.getcwd(), 'valid.txt')
    APs = {}
    # for i, cls in enumerate(self.classes):
    for i, cls in enumerate(classes_pred):  # clw modify
        R, P, AP = voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh, use_07_metric)
        APs[cls] = AP
    if os.path.exists(cachedir):
         shutil.rmtree(cachedir)

    mAP = 0
    for i in APs:
        print("{} --> mAP : {}".format(i, APs[i]))
        mAP += APs[i]
    mAP = mAP / len(classes_pred)
    print('mAP:%g' % (mAP))

    return APs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/voc_yolov3.cfg', help='xxx.cfg file path')
    #parser.add_argument('--cfg', type=str, default='cfg/voc_yolov3-spp.cfg', help='xxx.cfg file path')
    #parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='xxx.cfg file path')
    parser.add_argument('--data', type=str, default='cfg/voc.data', help='xxx.data file path')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='0,1', help='device id (i.e. 0 or 0,1,2,3) ') # 默认单卡
    parser.add_argument('--src-txt-path', type=str, default='./valid.txt', help='saved img_file_paths list')
    parser.add_argument('--dst-path', type=str, default='./output', help='save detect result in this folder')
    #parser.add_argument('--weights', type=str, default='weights/last.pt', help='path to weights file')
    parser.add_argument('--weights', type=str, default='weights/20200706_multiscale/last.pt', help='path to weights file')
    #parser.add_argument('--weights', type=str, default='weights/yolov3-spp.pt', help='path to weights file')
    parser.add_argument('--img-size', type=int, default=416, help='resize to this size square and detect')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold for compute mAP')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        test(opt.cfg,
             opt.data,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.nms_thres,
             opt.src_txt_path,
             opt.dst_path,
             opt.weights
             )