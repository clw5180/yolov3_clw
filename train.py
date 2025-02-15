### 数据集默认使用 voc2007+2012
### 1、训练速度：
# 1、单卡2080Ti
# （1）img_size=416 bs=32 一个batch用时 0.42s
# 2、多卡2080Ti x 2
# （1）img_size=416 bs=64 一个batch用时 0.56s，大概是 bs=32用时的1.3倍，但是batch总数只有原来的0.5倍，实测 1个epoch用时是单卡的 0.7倍

### 2、超参数调节 (这些重要的超参，都应该记录在 log内；
# （1）compute_loss  坐标（MSE，giou）
# （2）输入尺寸
#  (3) 预训练模型
#  (4) 学习率（lr、SGD/Adam、cosine/multi-step）、weight_decay、momentum、损失系数
#  (5) batchsize
#  (6) hyp['iou_t']  （ 在 build_targets()内  ）
#  (7) 数据增强方式（LetterBox，Resize）

# 腾讯优图的Yolov3用的和coco一样的anchor，32*range(10,19)也就是320~608多尺度训练，在544尺寸下测试可以达到mAP 79.6%；Yolov2达到mAP77.6%；
# Yolov2论文里给出的416尺寸下测试是75.4%，加多尺度训练是76.8%，然后再提高到544尺寸下测试是78.6%；所以个人感觉Yolov3在416尺寸下应该能达到接近77.4的水平；
# 腾讯优图的anchor策略有一点不同：对于某个yololayer，只要有某个anchor和gt的iou大于0.5，那么其他的anchor即使iou很小，也不作为负样本；
#  mask = (iou_gt_pred > self.thresh).sum(0) >= 1
#  conf_neg_mask[b][mask.view_as(conf_neg_mask[b])] = 0

# C语言原版yolov3中，使用的一些数据增强包括：
#（1）jitter=.3  ，利用数据抖动产生更多数据，属于TTA（Test Time Augmentation）
#（2）hsv：saturation = 1.5 ， exposure = 1.5 ，hue=.1
#（3）：learning_rate=0.001
# burn_in=1000
# max_batches = 50200
# policy=steps
# steps=40000,45000

######

import torch
import numpy as np
import cv2
from utils import custom_lr_scheduler

# Set printoptions
# torch.set_printoptions(linewidth=320, precision=5, profile='long')
# np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
# cv2.setNumThreads(0)
######

import argparse
from model.models import Darknet
from utils.utils import select_device, init_seeds, plot_images, plot_images2
from utils.parse_config import parse_data_cfg
import torch
import torch.optim.lr_scheduler as lr_scheduler
from dataset.datasets import VocDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import compute_loss, load_darknet_weights, write_to_file, weights_init_normal
import os
import time
import test
import test2
import torch.nn as nn
import torch.distributed as dist  #   clw note: TODO
from torch.utils.tensorboard import SummaryWriter
from utils.globals import log_file_path, log_folder, model_save_path
from utils.custom_lr_scheduler import adjust_learning_rate
import math
import random
import torch.nn.functional as F


### 超参数
#lr0 = 1e-3  # 太大，在前几个epoch的mAP反而并不好；在kaggle小麦数据集上损失甚至会溢出；
lr0 = 1e-4
momentum = 0.9
weight_decay = 0.0005
###

### 混合精度训练 ###
mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Waring: No Apex !!! ')  # https://github.com/NVIDIA/apex
    write_to_file('Waring: No Apex !!! ', log_file_path)
    mixed_precision = False        # not installed
if mixed_precision:
    print('Using Apex !!! ')
    write_to_file('Using Apex !!! ', log_file_path)
######

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--cfg', type=str, default='cfg/CSPDarknet53-PANet-SPP.cfg', help='xxx.cfg file path')
    # parser.add_argument('--cfg', type=str, default='cfg/resnet18.cfg', help='xxx.cfg file path')
    # parser.add_argument('--cfg', type=str, default='cfg/resnet50.cfg', help='xxx.cfg file path')
    # parser.add_argument('--cfg', type=str, default='cfg/resnet101.cfg', help='xxx.cfg file path')
    # parser.add_argument('--cfg', type=str, default='cfg/voc_yolov3-spp.cfg', help='xxx.cfg file path')
    parser.add_argument('--cfg', type=str, default='cfg/voc_yolov3.cfg', help='xxx.cfg file path')
    #parser.add_argument('--cfg', type=str, default='cfg/wheat_yolov3-spp.cfg', help='xxx.cfg file path')
    parser.add_argument('--data', type=str, default='cfg/voc.data', help='xxx.data file path')
    #parser.add_argument('--data', type=str, default='cfg/wheat.data', help='xxx.data file path')
    parser.add_argument('--device', default='0', help="device id (i.e. 0 or 0,1,2,3) or '' ") # 如果为空，则默认使用所有当前可用的显卡
    #parser.add_argument('--weights', type=str, default='weights/cspdarknet53-panet-spp.weights', help='path to weights file')
    # parser.add_argument('--weights', type=str, default='weights/resnet18.pth', help='path to weights file')
    # parser.add_argument('--weights', type=str, default='weights/resnet50.pth', help='path to weights file')
    #parser.add_argument('--weights', type=str, default='weights/resnet101.pth', help='path to weights file')
    #parser.add_argument('--weights', type=str, default='weights/yolov3-spp.pt', help='path to weights file')
    #parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='path to weights file')
    #parser.add_argument('--weights', type=str, default='weights/yolov3.pt', help='path to weights file')
    #parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--weights', type=str, default='weights/darknet53.conv.74', help='path to weights file')
    #parser.add_argument('--weights', type=str, default='', help='path to weights file')
    #parser.add_argument('--weights', type=str, default='', help='path to weights file')
    #parser.add_argument('--img-size', type=int, default=1024, help='resize to this size square and detect')
    parser.add_argument('--img-size', type=int, default=416, help='resize to this size square and detect')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    #parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    #parser.add_argument('--batch-size', type=int, default=8)
    #parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--multi-scale', action='store_true', help='rectangular training')
    opt = parser.parse_args()

    # 0、参数设置 ( 设置随机数种子，读取参数和配置文件信息 )
    cfg = opt.cfg
    weights = opt.weights
    img_size = opt.img_size
    batch_size = opt.batch_size
    total_epochs = opt.epochs
    init_seeds(0)
    data = parse_data_cfg(opt.data)
    train_txt_path = data['train']
    valid_txt_path = data['valid']
    nc = int(data['classes'])
    device = select_device(opt.device)

    # 打印配置信息，写log等
    print(opt)
    print('config file:', cfg)
    print('pretrained weights:', weights)
    print('initial lr:', lr0)

    os.makedirs(log_folder, exist_ok=True)
    write_to_file(repr(opt), log_file_path)
    write_to_file('config file:' + cfg, log_file_path)
    write_to_file('pretrained weights:' + repr(weights), log_file_path)
    write_to_file('initial lr:' + repr(lr0), log_file_path)


    # 1、加载模型
    model = Darknet(cfg).to(device)
    model.apply(weights_init_normal)  # clw note: without this can also get high mAP;   TODO

    print('anchors:\n' + repr(model.module_defs[model.yolo_layers[0]]['anchors']))
    write_to_file('anchors:\n' + repr(model.module_defs[model.yolo_layers[0]]['anchors']), log_file_path)

    if weights.endswith('.pt'):
        ### model.load_state_dict(torch.load(weights)['model']) # 错误原因：没有考虑类别对不上的那一层，也就是yolo_layer前一层
                                                                # 会报错size mismatch for module_list.81.Conv2d.weight: copying a param with shape torch.size([255, 1024, 1, 1]) from checkpoint, the shape in current model is torch.Size([75, 1024, 1, 1]).
        chkpt = torch.load(weights)
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s" % (opt.weights, opt.cfg)
            raise KeyError(s) from e

    elif weights.endswith('.pth'):    # for 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        model_state_dict = model.state_dict()
        chkpt = torch.load(weights, map_location=device)
        state_dict = {}
        block_cnt = 0
        fc_item_num = 2
        chkpt_keys = list(chkpt.keys())
        model_keys = list(model.state_dict().keys())
        model_values = list(model.state_dict().values())
        for i in range(len(chkpt_keys) - fc_item_num):  # 102 - 2
            if i % 5 == 0:
                state_dict[model_keys[i+block_cnt]] = chkpt[chkpt_keys[i]]
            elif i % 5 == 1 or i % 5 == 2:
                state_dict[model_keys[i+block_cnt+2]] = chkpt[chkpt_keys[i]]
            elif i % 5 == 3 or i % 5 == 4:
                state_dict[model_keys[i+block_cnt-2]] = chkpt[chkpt_keys[i]]
                if i % 5 == 4:
                    block_cnt += 1
                    state_dict[model_keys[i + block_cnt]] = model_values[i + block_cnt]

        #chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(state_dict, strict=False)
        # model.load_state_dict(chkpt['model'])
        # except KeyError as e:
        #     s = "%s is not compatible with %s" % (opt.weights, opt.cfg)
        #     raise KeyError(s) from e

    elif len(weights) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        load_darknet_weights(model, weights)

    # else:
    #     raise Exception("pretrained model's path can't be NULL!")

    # 2、加载数据集
    train_dataset = VocDataset(train_txt_path, img_size, with_label=True, is_training=True)
    dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8, # TODO
                            collate_fn=train_dataset.train_collate_fn,
                            pin_memory=True)

    # 3、设置优化器 和 学习率
    start_epoch = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=momentum, weight_decay=weight_decay)  # TODO:  nesterov=True
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr0)  # TODO: can't use Adam

    ######
    # # Optimizer
    # pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    # for k, v in dict(model.named_parameters()).items():
    #     if '.bias' in k:
    #         pg2 += [v]  # biases
    #     elif 'Conv2d.weight' in k:
    #         pg1 += [v]  # apply weight_decay
    #     else:
    #         pg0 += [v]  # parameter group 0
    #
    # optimizer = torch.optim.SGD(pg0, lr=lr0, momentum=momentum, nesterov=True)
    #
    # clw note: 使用add_param_group(),可以实现比如不同层使用不同的lr和weight decay
    # optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})  # add pg1 with weight_decay
    # optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # del pg0, pg1, pg2
    ######


    #### 阶梯学习率
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(total_epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    ### 余弦学习率
    # lf = lambda x: (1 + math.cos(x * math.pi / total_epochs)) / 2
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    ### 余弦学习率2 - batch step
    scheduler = custom_lr_scheduler.CosineDecayLR(optimizer,
                                                T_max=total_epochs * len(dataloader),
                                                lr_init=lr0,
                                                lr_min=lr0 * 1e-3,
                                                warmup=5 * len(dataloader))
    ### 阶梯学习率2：参考腾讯优图yolov3
    # lr_steps: [400,700,900,1000, 40000,45000]
    # lr_rates: [0.0001,0.0002,0.0005,0.001, 0.0001,0.00001]


    ###### apex need ######
    if device == 'cpu':
        mixed_precision = False
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
    # Initialize distributed training
    #if torch.cuda.device_count() > 1:
    if len((opt.device).split(',')) > 1:  # clw modify
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)  # clw note: 多卡,在 amp.initialize()之后调用分布式代码 DistributedDataParallel否则报错
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level
    ######
    model.nc = nc
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing

    multi_scale = opt.multi_scale
    if multi_scale:
        print('using multi_scale !')
        write_to_file('using multi_scale !', log_file_path)

    # 4、训练
    print('')   # 换行
    print('Starting training for %g epochs...' % total_epochs)
    time0 = time.time()
    nb = len(dataloader)


    writer = SummaryWriter()    # tensorboard --logdir=runs, view at http://localhost:6006/

    prebias = start_epoch == 0

    for epoch in range(start_epoch, total_epochs):  # epoch ------------------------------
        model.train()  # 写在这里，是因为在一个epoch结束后，调用test.test()时，会调用 model.eval()
        mloss = torch.zeros(4).to(device)  # mean losses
        # # Prebias
        # if prebias:
        #     if epoch < 3:  # prebias
        #         ps = 0.1, 0.9  # prebias settings (lr=0.1, momentum=0.9)
        #     else:  # normal training
        #         ps = lr0, momentum  # normal training settings
        #         print_model_biases(model)
        #         prebias = False
        #
        #     # Bias optimizer settings
        #     optimizer.param_groups[2]['lr'] = ps[0]
        #     if optimizer.param_groups[2].get('momentum') is not None:  # for SGD but not Adam
        #         optimizer.param_groups[2]['momentum'] = ps[1]

        start = time.time()
        title = ('\n' + '%10s' * 11 ) % ('Epoch', 'Batch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size', 'lr', 'time_use')
        print(title)
        #pbar = tqdm(dataloader, ncols=20)  # 行数参数ncols=10，这个值可以自己调：尽量大到不能引起上下滚动，同时满足美观的需求。
        #for i, (img_tensor, target_tensor, img_path, _) in enumerate(pbar):

        # # Freeze darknet53.conv.74 for first epoch
        # freeze_backbone = False
        # if freeze_backbone and (epoch < 3):
        #     for i, (name, p) in enumerate(model.named_parameters()):
        #         if int(name.split('.')[2]) < 75:  # if layer < 75  # 多卡是[2]因为是 model.module.xxx，单卡[1]
        #             p.requires_grad = False if (epoch < 3) else True


        for i, (img_tensor, target_tensor, img_path) in enumerate(dataloader):

            # 调整学习率，进行warm up和学习率衰减
            ## Update scheduler per batch

            # clw note: SGD burn-in is very important when starting from stratch or only load darknet53.conv.74,
            #           because it's easy to cause loss infinite
            # ni = epoch * nb + i
            # if ni <= 500:  # n_burnin = 1000
            #     lr = lr0 * (ni / 500) ** 2
            #     for g in optimizer.param_groups:
            #         g['lr'] = lr

            ni = epoch * nb + i
            scheduler.step(ni)
            # lr = adjust_learning_rate(optimizer, 0.1, lr0, total_epochs, epoch, ni, nb)

            batch_start = time.time()
            img_tensor = img_tensor.to(device)
            target_tensor = target_tensor.to(device)


            ######
            # Multi-Scale training
            if multi_scale:
                if ni  % 10 == 0:  #  adjust (67% - 150%) every 1 or 10 batches
                    img_size = random.randrange(10, 19) * 32  # 320~608, 间隔32
                    #img_size = random.randrange(24, 32) * 32
                ##ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                img_tensor = F.interpolate(img_tensor, size=(img_size, img_size), mode='bilinear', align_corners=False)

            img_size = img_tensor.size()[2]  # TODO: 目前只支持正方形，如416x416
            ######

            ### 训练过程主要包括以下几个步骤：
            # (1) 前传
            #print('img_tensor:', img_tensor[0][1][208][208])
            p, p_box = model(img_tensor)  # tuple, have 3 tensors; tensor[0]: (64, 3, 13, 13, 4)

            # (2) 计算损失
            ######  clw add: for debug, localize in build_target() first, and can get target size, so catch the same target size there
            # if target_tensor.size()[0] == 679:
            #     print('aaa')
            ######
            loss, loss_items = compute_loss(p, p_box, target_tensor, model, img_size)
            if not torch.isfinite(loss):
               raise Exception('WARNING: non-finite loss, ending training ', loss_items)

            # (3) 损失：反向传播，求出梯度
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # (4) 优化器：更新参数、梯度清零
            ni = i + nb * epoch  # number integrated batches (since train start)
            if ni % accumulate == 0:  # Accumulate gradient for x batches before optimizing
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            #s = ('%10s' * 2 + '%10.3g' * 7 + '%10.3gs') % ('%g/%g' % (epoch, total_epochs - 1), '%.3gG' % mem, *mloss, len(target_tensor), img_size,  scheduler.get_lr()[0], time.time()-batch_start)

            #s = ('%10s' * 3 + '%10.3g' * 7 + '%10.3gs') % ('%g/%g' % (epoch, total_epochs - 1), '%g/%g' % (i, nb - 1), '%.3gG' % mem, *mloss, len(target_tensor), img_size,  optimizer.state_dict()['param_groups'][0]['lr'], time.time()-batch_start)
            s = ('%10s' * 3 + '%10.3g' * 7 + '%10.3gs') % ('%g/%g' % (epoch, total_epochs - 1), '%g/%g' % (i, nb - 1), '%.3gG' % mem, *mloss, len(target_tensor), img_size,  optimizer.param_groups[0]['lr'], time.time()-batch_start)
            #s = ('%10s' * 3 + '%10.3g' * 7 + '%10.3gs') % ('%g/%g' % (epoch, total_epochs - 1), '%g/%g' % (i, nb - 1), '%.3gG' % mem, *mloss, len(target_tensor), img_size,  scheduler.get_lr()[0], time.time()-batch_start)


            if i % 10 == 0:
                print(s)
                
            # Plot
            if epoch == start_epoch  and i == 0:
                fname = 'train_batch0.jpg' # filename
                fname2 = 'train_batch00.jpg'
                cur_path = os.getcwd()
                plot_images2(img_tensor, target_tensor, paths=img_path, fname=os.path.join(cur_path, fname2))
                res = plot_images(images=img_tensor, targets=target_tensor, paths=img_path, fname=os.path.join(cur_path, fname))
                # writer.add_image(fname, res, dataformats='HWC', global_step=epoch)
                # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        print('time use per epoch: %.3fs' % (time.time() - start))

        write_to_file(title, log_file_path)
        write_to_file(s, log_file_path)

        ### Update scheduler per epoch
        # scheduler.step()

        # compute mAP
        if epoch >= 3 and (epoch+1) % 5 == 0:  # clw note: avoid nms cause too much time for epoch1 and epoch2
        #if epoch >= 3:  # clw note: avoid nms cause too much time for epoch1 and epoch2
            results, maps = test.test(cfg,
                      opt.data,
                      batch_size=batch_size,
                      img_size=img_size,
                      conf_thres=0.1,
                      iou_thres=0.5,
                      nms_thres=0.5,
                      src_txt_path=valid_txt_path,
                      weights=None,
                      log_file_path=log_file_path,
                      model=model)

            # save model   TODO
            last_chkpt = {'epoch': epoch,
                          'model': model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel else model.state_dict(),  # clw note: 多卡
                          'optimizer': optimizer.state_dict()}
            torch.save(last_chkpt, model_save_path)

            # Delete checkpoint
            del last_chkpt

    print('Training end! total time use: %.3fs' % (time.time() - time0))





