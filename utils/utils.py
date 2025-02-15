import os
import math
import torch
import torchvision
import numpy as np
import cv2
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.globals import log_file_path, pytorch_version_minor

import time
####################


### clw note: anchor匹配策略如下（和原论文相同）：每一层都有且只有一个anchor和某个gt匹配
# (1) 每一个 yololayer都找到最大的iou的anchor，作为正样本
# (2) 小于最大iou，但大于iou_thres忽略
# (3) 小于iou_thres作为负样本
def build_targets(model, bs, targets):  # targets = (n, 6)， 每一行形如 [image_idx, class, x, y, w, h]

    ByteTensor = torch.cuda.ByteTensor if targets.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if targets.is_cuda else torch.FloatTensor

    nc = model.nc
    nt = len(targets)  #  nt: number of targets，比如64个batch，每个batch也就是每张图有3个gt，则一共有192个targets# 有gt，才能往后计算gt和iou匹配的过程，选出进入compute loss的gt

    tcls_all = []
    tx_all, ty_all, th_all, tw_all = [], [], [], []
    obj_mask_all, noobj_mask_all = [], []
    target_all = []  # clw note: 在compute_loss中转化为gt，然后计算pred和gt的iou，大于0.5的计算 noobj损失
    mixup_ratios_all = []

    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    for i in model.yolo_layers:  # clw note: 一层一层来

        # get number of grid points and anchor vec for this yolo layer
        if multi_gpu:
            ng, anchor_vec = model.module.module_list[i].ng[0], model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng[0], model.module_list[i].anchor_vec
        na = len(anchor_vec)  # number of anchors:3

        ### Output tensors
        obj_mask = ByteTensor(bs, na, ng, ng).fill_(0)
        noobj_mask = ByteTensor(bs, na, ng, ng).fill_(1)
        tx = FloatTensor(bs, na, ng, ng).fill_(0)
        ty = FloatTensor(bs, na, ng, ng).fill_(0)
        tw = FloatTensor(bs, na, ng, ng).fill_(0)
        th = FloatTensor(bs, na, ng, ng).fill_(0)
        tmixup_ratios = FloatTensor(bs, na, ng, ng).fill_(1)
        tcls = FloatTensor(bs, na, ng, ng, nc).fill_(0)
        ###

        ### Convert to position relative to box
        gwh = targets[:, 4:6] * ng  # (n, 2)
        gxy = targets[:, 2:4] * ng  # grid x, y
        gmixup_ratios = targets[:, 6]
        ###

        # iou = torch.stack([wh_iou(x, gwh) for x in anchor_vec], 0)
        #ious = torch.stack([bboxes_anchor_iou( torch.cat((gxy, gwh), 1), anchor, x1y1x2y2=False) for anchor in anchor_vec], 0)  # clw modify: wh_iou is not accurate enough   (3, n)

        ### Get anchors with best iou
        ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchor_vec])
        best_ious, best_iou_mask = ious.max(0)  # 三个anchor里面找最大的  iou_mask: (263,)

        ### Separate target values
        batch_idx, class_id = targets[:, :2].long().t()  # target image, class
        gi, gj = gxy.long().t()  # 整数，表示target所在grid的索引，形如(x, y)
        gx, gy = gxy.t()
        gw, gh = gwh.t()

        ### Set masks, and Set noobj mask to zero where iou exceeds ignore threshold
        obj_mask[batch_idx, best_iou_mask, gj, gi] = 1
        noobj_mask[batch_idx, best_iou_mask, gj, gi] = 0
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[batch_idx[i], anchor_ious > 0.5, gj[i], gi[i]] = 0 # 本来noobj全写1，然后最大iou那个写0，然后再把iou大于0.5的全部写0，这样可以达到论文的效果；
        obj_mask_all.append(obj_mask)
        noobj_mask_all.append(noobj_mask)
        ###

        # 在上面筛选出和anchor的iou符合要求的那些gt之后，把这些gt记录下来，作为compute_loss() 希望回归的目标；
        # 并且转化为回归时所需要的格式，比如实际要回归的是cell的偏差 gx - gx.floor()，而不是gt的实际坐标 gx
        tx[batch_idx, best_iou_mask, gj, gi] = gx - gx.floor()  # TODO: if there is the situation that some anchor match 2 gt, the anchor will match the last
        ty[batch_idx, best_iou_mask, gj, gi] = gy - gy.floor()  #
        tw[batch_idx, best_iou_mask, gj, gi] = torch.log(gw / anchor_vec[best_iou_mask][:, 0] + 1e-16)
        th[batch_idx, best_iou_mask, gj, gi] = torch.log(gh / anchor_vec[best_iou_mask][:, 1] + 1e-16)
        tmixup_ratios[batch_idx, best_iou_mask, gj, gi] = gmixup_ratios
        # tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        tx_all.append(tx)
        ty_all.append(ty)
        tw_all.append(tw)
        th_all.append(th)
        mixup_ratios_all.append(tmixup_ratios)
        # Class
        tcls[batch_idx, best_iou_mask, gj, gi, class_id] = 1
        tcls_all.append(tcls)

        if class_id.shape[0]:  # if any targets
            assert class_id.max() <= model.nc, 'Model accepts %g classes labeled from 0-%g, however you supplied a label %g. See \
                                        https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
            model.nc, model.nc - 1, class_id.max())

    return obj_mask_all, noobj_mask_all, tx_all, ty_all, tw_all, th_all, tcls_all, target_all, mixup_ratios_all


'''
### clw note: anchor匹配策略如下：
# (1) find max iou anchor in all !! yololayer, not one yololayer
# (2) split all max iou anchors and Gts into 3 yololayer, send to compute_loss()
def build_targets(model, targets):  # targets: (n, 6)， 每一行形如 [image_idx, class, x, y, w, h]
    nt = len(targets)   #  nt: number of target
    tcls, tbox, indices, av = [], [], [], []
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)  # clw note: 多卡
    iou_all = torch.zeros((9, nt)).cuda()    # (9, n)  # TODO:这里9=3*3还要改一下
    a_all = torch.zeros((9*nt)).long().cuda()  # clw modify
    t_all = torch.zeros((9*nt, 6)).cuda()
    gwh_all = torch.zeros((9*nt, 2)).cuda()
    for yololayer_idx, i in enumerate(model.yolo_layers):  # clw note: 遍历3个yolo层
        # get number of grid points and anchor vec for this yolo layer
        if multi_gpu:
            ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec
            # ng=[13, 13]  anchor_vec=([[ 3.6250,  2.8125],[ 4.8750,  6.1875], [11.6562, 10.1875]]
            # anchor 在yololayer forward创建， 是除以stride之后的值, 如 116/32=3.625, 90/32=2.8125...

        # iou of targets-anchors
        gwh = targets[:, 4:6] * ng  # gt换算到三个feature map的 wh
        gxy_clw = targets[:, 2:4] * ng  # grid x, y
        gxywh = torch.cat((gxy_clw, gwh), 1)
        if nt:
            ############################
            # use_all_anchors, reject = True, True
            # iou = torch.stack([wh_iou(x, gwh) for x in anchor_vec], 0)
            # if use_all_anchors:
            #     na = len(anchor_vec)  # number of anchors
            #     a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)
            #     t = targets.repeat([na, 1])
            #     gwh = gwh.repeat([na, 1])
            #     iou = iou.view(-1)  # use all ious
            # else:  # use best anchor only
            #     iou, a = iou.max(0)  # best iou and anchor
            #
            # # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            # if reject:
            #     j = iou.view(-1) > 0.2  # iou threshold hyperparameter
            #     #j = iou > model.hyp['iou_t']  # iou threshold hyperparameter
            #     t, a, gwh = t[j], a[j], gwh[j]
            #############################
            na = len(anchor_vec)  # number of anchors
            a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1).long()
            #a = torch.full((na, nt), yololayer_idx).long().cuda()  # (3, 5)
            t = targets.repeat([na, 1])  # (5*3, 6)
            gwh = gwh.repeat([na, 1])    # (2*3, 6)

            a_all[( a.size()[0] * yololayer_idx):(a.size()[0] * (yololayer_idx + 1))] = a
            t_all[(t.size()[0] * yololayer_idx):(t.size()[0] * (yololayer_idx + 1)), :] = t  # (45, 6)
            gwh_all[(gwh.size()[0] * yololayer_idx):(gwh.size()[0] * (yololayer_idx + 1)), :] = gwh
            iou = torch.stack([bboxes_anchor_iou(gxywh, anchor, x1y1x2y2=False) for anchor in anchor_vec], 0)
            iou_all[(iou.size()[0] * yololayer_idx):(iou.size()[0] * (yololayer_idx + 1)), :] = iou          # clw modify

    iou, j = iou_all.max(0)  # (9, nt) ->  (1, nt)  这里 nt=7

    # mask = torch.zeros(iou_all.size(), dtype=torch.bool)
    # for i in range(len(j)):  # TODO: 不用for循环
    #     mask[j[i], i] = 1
    # mask = mask.t().reshape(1, -1).squeeze()
    # # best iou and anchor  # 在这一层的 3个anchor和所有 target的 iou里面找一个最大的
    # # #返回每一列中最大值的那个元素，且返回索引（返回最大元素在这一列的行索引）
    # t_all, a_all, gwh_all = t_all[mask], a_all[mask], gwh_all[mask]  # (nt, 6)   (nt)   (nt, 2)


    for yololayer_idx, i in enumerate(model.yolo_layers):
        if multi_gpu:
            ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec

        mask_yololayer_idx = (a_all == yololayer_idx)  # clw note: compute respectively in 3 yolo_layers: [0, 1, 2], which is anchor_idx
        targets, a, gwh = t_all[mask_yololayer_idx], a_all[mask_yololayer_idx], gwh_all[mask_yololayer_idx]

        # Indices
        b, c = targets[:, :2].long().t()  # target image idx, class
        gxy = targets[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))  # 加入的4个元素是：image_idx(0~batchsize), anchor_idx(0~2), gt的x_ctr和y_ctr所在cell左上角坐标(范围依次是0~12, 0~25, 0~51)，整数

        # GIoU
        gxy -= gxy.floor()  # gt的x_ctr和y_ctr所在cell 偏移量   # TODO !!!
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)， concat后变成 (n, 2+2)
        av.append(anchor_vec[a])  # anchor vec，就是iou超过threshold的 anchor index

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() < model.nc, 'Target class_idx exceed total model classes'

    return tcls, tbox, indices, av
'''



'''
    for idx, i in enumerate(model.yolo_layers):
        if multi_gpu:
            ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec

        # j = (iou_all == iou_max)  # | (iou_all < 0.3)   # TODO: 两个iou相同的情况
        targets, a, gwh = targets[j], a[j], gwh[j]

        # Indices
        b, c = targets[:, :2].long().t()  # target image idx, class
        gxy = targets[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))  # 加入的4个元素是：image_idx(0~batchsize), anchor_idx(0~2), gt的x_ctr和y_ctr所在cell左上角坐标(范围依次是0~12, 0~25, 0~51)，整数

        # GIoU
        gxy -= gxy.floor()  # gt的x_ctr和y_ctr所在cell 偏移量   # TODO !!!
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)， concat后变成 (n, 2+2)
        av.append(anchor_vec[a])  # anchor vec，就是iou超过threshold的 anchor index

        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            assert c.max() < model.nc, 'Target class_idx exceed total model classes'

    return tcls, tbox, indices, av  # t前缀表示target，也就是gt，tbox: (n, 4) 其中 n 代表iou超过threshold的 anchor个数，
                                    # 比如n=2且只有1个target，那么2行的内容都是一样的，意思是该层yolo_layer有2个anchor命中该target

                                    # tcls是一个list，[0][1][2]分别为3个层对应的3个tensor，每个tensor:(n)，记录了每个 gt 对应的 cls
                                    # indices是一个list，[0][1][2]分别为3个层对应的3个tuple，每个tuple包含4个tensor，每个tensor:(n)，记录了 image_idx(0~batchsize), anchor_idx(0~2), gt的x_ctr和y_ctr所在cell左上角坐标(范围依次是0~12, 0~25, 0~51)，整数
                                    # tbox是一个list，[0][1][2]分别为3个层对应的3个tensor, 每个tensor:(n, 4)，其中4个值为xywh，xy是当前grid cell内的坐标，wh是gt映射到当前feature map的wh
                                    # av也是一个list，[0][1][2]分别为3个层对应的3个tensor，每个tensor:(n, 2)，如 [3.625, 2.8125]，记录了每一层和每个gt 匹配的 anchor的w和h
'''



'''
### clw note: anchor匹配策略如下：
#（1）anchor和gt的iou大于0.3的均视为正样本
#（2）如果所有anchor和gt的iou都小于0.3，则跳过
def build_targets(model, bs, targets):   # build mask Matrix according to batchsize
    # targets = [image, class, x, y, w, h]
    ByteTensor = torch.cuda.ByteTensor if targets.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if targets.is_cuda else torch.FloatTensor

    nc = model.nc
    nt = len(targets)  # 有gt，才能往后计算gt和iou匹配的过程，选出进入compute loss的gt

    tcls_all = []
    tx_all, ty_all, th_all, tw_all = [], [], [], []
    obj_mask_all, noobj_mask_all = [], []
    target_all = []  # clw note: 在compute_loss中转化为gt，然后计算pred和gt的iou，大于0.5的计算 noobj损失
    mixup_ratios_all = []

    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    for i in model.yolo_layers:  # clw note: 一层一层来

        # get number of grid points and anchor vec for this yolo layer
        if multi_gpu:
            ng, anchor_vec = model.module.module_list[i].ng[0], model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng[0], model.module_list[i].anchor_vec

        na = len(anchor_vec)
        obj_mask = ByteTensor(bs, na, ng, ng).fill_(0)   # clw note: if nt's range from (0~63), the batch_size is 64.
        noobj_mask = ByteTensor(bs, na, ng, ng).fill_(1)
        tx = FloatTensor(bs, na, ng, ng).fill_(0)
        ty = FloatTensor(bs, na, ng, ng).fill_(0)
        tw = FloatTensor(bs, na, ng, ng).fill_(0)
        th = FloatTensor(bs, na, ng, ng).fill_(0)
        tmixup_ratios = FloatTensor(bs, na, ng, ng).fill_(1)
        tcls = FloatTensor(bs, na, ng, ng, nc).fill_(0)

        # iou of targets-anchors
        a = []
        targets_cur = targets
        gwh = targets[:, 4:6] * ng  # (171, 2)
        gxy = targets[:, 2:4] * ng  # grid x, y
        gxywh = torch.cat((gxy, gwh), 1)  # clw note: TODO
        gmixup_ratios = targets[:, 6]

        if nt:
            #iou = torch.stack([wh_iou(x, gwh) for x in anchor_vec], 0)
            iou = torch.stack([bboxes_anchor_iou(gxywh, anchor, x1y1x2y2=False) for anchor in anchor_vec], 0)  # clw modify: wh_iou is not accurate enough
            iou_mask = (iou > 0.3)
            targets_need = targets[ iou_mask.sum(0) > 0 ]
            ###### clw modify
            gts = []  # list, len: bs
            pre_batch_idx = 0
            gt_box = []
            for batch_idx, target in enumerate(targets_need):
                if targets_need[batch_idx][0] == pre_batch_idx:  # 相同batch_idx，直接进box_list
                    gt_box.append(target[2:])
                else:  # 不同batch_idx，
                    if targets_need[batch_idx][0] - pre_batch_idx >= 1:  # 分两种情况：（1）相差等于1，很正常 （2）大于1，说明某个batch没有iou匹配的gt，
                                                                         #                   此时就要把空缺的batch补上[]，这样确保维度始终是batchsize
                        pre_batch_idx += 1
                        gts.append(gt_box) # 先将之前的box_list进总的list
                        gt_box = []  # 同时box_list清零，
                        gt_box.append(target[2:])  # 然后当前这个box进box_list
                        while(targets_need[batch_idx][0] - pre_batch_idx > 0): # 如果相差大于0，也就是上面的情况（2），那么重复操作以下两步；相当于比如batch_idx=3的时候，一定要把batch_idx为0,1,2的所有gt都处理完
                            gts.append([])  # 加一个空的box_list到总的list
                            pre_batch_idx += 1
                        continue  # 相差等于0，回到循环开始，继续往下找
                    else:  # 相差小于0，说明不是targets_need不是按照第一个维度也就是batch_idx顺序排列的，因此有bug
                        raise Exception('Error: targets_need[batch_idx][0] - pre_batch_idx < 1 never occur! maybe have some bug')
            gts.append(gt_box)  # 把最后留下的同一batch_idx的gt加进去; gts looks like:  list[0]: [tensor, tensor....]  list[1]: [tensor], ... , list[63]: [tensor, ...]

            if targets_need.numel():  # 非空，有具体的元素存在，而不是[]这种元素
                for i in range(bs - 1 - int(target[0])):  # clw note: 比如bs=64, 但是batch_idx=60以后都没有匹配的target，那么这里要append 3个[]
                    gts.append([])
            else:   # 否则 target可能还是上一个layer遗留下来的东西，而不是targets_need的元素！！ 因为targets_need是空的....
                for i in range(bs - 1): # 因此就要加上bs-1个空的列表
                    gts.append([])


            # 遍历64个batch，找到一个batch含有的gt数量最多是多少，将该值写入gt_max_len
            # print('clw: len(gts) =', len(gts))  # for debug; if not equal to the batchsize(except last batch in epoch), or the 3 layer values are different, it means bug !!
            gt_max_len = -1
            for batch_idx in range(len(gts)):
                if len(gts[batch_idx]) > gt_max_len:
                    gt_max_len = len(gts[batch_idx])
            gt_all = torch.cuda.FloatTensor(len(gts), gt_max_len ,5).fill_(0)   # 5 -> (x, y, w, h, mixup_ratio)

            for batch_idx in range(len(gts)):
                for target_idx in range(len(gts[batch_idx])):
                    #print(batch_idx, target_idx)
                    gt_all[batch_idx, target_idx] = gts[batch_idx][target_idx]

            ###### clw note: can debug
            target_all.append(gt_all)  # gt_all: tensor(bs, n, 4), n is max gt num in batchsize images, such as 150 in Peterisfer
            ######

            na = len(anchor_vec)  # number of anchors
            a = torch.arange(na).view((-1, 1)).repeat([1, nt]).view(-1)  # anchor_idx, represent which anchor in this yolo_layer, 0, 1 or 2
            targets_cur = targets_cur.repeat([na, 1])
            gxy = gxy.repeat([na, 1])
            gwh = gwh.repeat([na, 1])
            gmixup_ratios = gmixup_ratios.repeat(na)

            ###### clw modify
            # if pytorch_version_minor <= 1:  # pytorch 1.1 or less
            #     iou_mask1 = torch.zeros(iou.size()).byte().cuda()
            # else:
            #     iou_mask1 = torch.zeros(iou.size()).bool().cuda()
            # _, idx = iou.max(0)
            # for i in range(len(idx)):  # clw note: for loop's efficiency is low  TODO
            #     iou_mask1[idx[i], i] = 1  #        torch.sum(): 201
            # iou_mask2 = (iou > 0.3)  # (3, 201)  torch.sum(): 305
            # iou_mask = iou_mask1 | iou_mask2  # torch.sum(): 330
            ######


            iou_mask = iou_mask.view(-1)
            targets_cur, a, gxy, gwh, gmixup_ratios = targets_cur[iou_mask], a[iou_mask], gxy[iou_mask], gwh[iou_mask], gmixup_ratios[iou_mask]


        # Indices
        batch_idx, class_id = targets_cur[:, :2].long().t()  # target image, class
        gi, gj = gxy.long().t()  # 整数，表示target所在grid的索引，形如(x, y)

        ################################################################################################
        # clw add: for debug the error -> RuntimeError: CUDA error: device-side assert triggered terminate called after throwing an instance of 'c10::Error' what():  CUDA error: device-side assert triggered
        # if ng == 13:
        #     error_mask = (gj[:] >= 13) | (gj[:] <0) | (gi[:] >= 13) | (gi[:] <0)
        #     if error_mask.sum() != 0:
        #         print('13')
        #         print(error_mask)
        # if ng == 26:
        #     error_mask = (gj[:] >= 26) | (gj[:] <0) | (gi[:] >= 26) | (gi[:] <0)
        #     if error_mask.sum() > 0:
        #         print('26')
        #         print(error_mask)
        # if ng == 52:
        #     error_mask = (gj[:] >= 52) | (gj[:] < 0) | (gi[:] >= 52) | (gi[:] <0)
        #     if error_mask.sum() > 0:
        #         print('52')
        #         print(error_mask)
        ################################################################################################

        #print(batch_idx, a, gj, gi)  # for debug
        obj_mask[batch_idx, a, gj, gi] = 1
        #aaa = torch.sum(obj_mask)  # TODO: aaa is 200, not 201, so some anchor match 2 gt
        noobj_mask[batch_idx, a, gj, gi] = 0

        # Set noobj mask to zero where iou exceeds ignore threshold
        # if not use_all_anchors:  # use best anchor, such as paper did
        #     for i, iou_ in enumerate(iou.t()):  # iou.t(): (201, 3)    iou_: (3,)
        #         noobj_mask[batch_idx[i], iou_ > 0.5, gj[i], gi[i]] = 0    # 0.5 is ignore_thres, such as paper said


        obj_mask_all.append(obj_mask)
        noobj_mask_all.append(noobj_mask)

        # 在上面筛选出和anchor的iou符合要求的那些gt之后，把这些gt记录下来，作为compute_loss() 希望回归的目标；
        # 并且转化为回归时所需要的格式，比如实际要回归的是cell的偏差 gx - gx.floor()，而不是gt的实际坐标 gx
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        tx[batch_idx, a, gj, gi] = gx - gx.floor()  # TODO: if there is the situation that some anchor match 2 gt, the anchor will match the last
        ty[batch_idx, a, gj, gi] = gy - gy.floor()  #
        tw[batch_idx, a, gj, gi] = torch.log(gw / anchor_vec[a][:, 0] + 1e-16)
        th[batch_idx, a, gj, gi] = torch.log(gh / anchor_vec[a][:, 1] + 1e-16)
        tmixup_ratios[batch_idx, a, gj, gi] = gmixup_ratios
        # tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        tx_all.append(tx)
        ty_all.append(ty)
        tw_all.append(tw)
        th_all.append(th)
        mixup_ratios_all.append(tmixup_ratios)

        # Class
        tcls[batch_idx, a, gj, gi, class_id] = 1
        tcls_all.append(tcls)

        if class_id.shape[0]:  # if any targets
            assert class_id.max() <= model.nc, 'Model accepts %g classes labeled from 0-%g, however you supplied a label %g. See \
                                        https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (model.nc, model.nc - 1, class_id.max())

    return obj_mask_all, noobj_mask_all, tx_all, ty_all, tw_all, th_all, tcls_all, target_all, mixup_ratios_all
'''


class MyFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):  # gamma: 控制难易样本，容易样本pt更接近1，则(1-pt)更接近0，平方则loss更小，降低容易样本起的作用；
                                                                 # alpha：控制政府样本；label=1时alpha=0.75, label=-1时alpha=0.25，降低负样本起的作用；
        super(MyFocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        pos_mask = (target == 1)
        neg_mask = (target == 0)
        #loss_pos = (1 - self.__alpha) * (torch.pow(torch.abs(target[pos_mask] - torch.sigmoid(input[pos_mask])), self.__gamma) *  self.__loss(input=input[pos_mask], target=target[pos_mask])).sum()
        #loss_neg = self.__alpha * (torch.pow(torch.abs(target[neg_mask] - torch.sigmoid(input[neg_mask])), self.__gamma) *  self.__loss(input=input[neg_mask], target=target[neg_mask])).sum()
        loss_pos = (1 - self.__alpha) * (torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma) *  self.__loss(input=input, target=target)) * pos_mask.float()  ## TODO：重复计算
        loss_neg = self.__alpha * (torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma) *  self.__loss(input=input, target=target)) * neg_mask.float()
        loss = loss_pos + loss_neg
        return loss

### new version
def compute_loss(p, p_box, targets, model, img_size):  # p:predictions，一个list包含3个tensor，维度(bs,3,13,13,25), (bs,3,26,26,25)....
                                             # p_box: 一个list包含3个tensor，维度(bs,3,13,13,4), (bs,3,26,26,4)....   targets: (n, 6)
    lcls, lbox, lobj = torch.cuda.FloatTensor([0]), torch.cuda.FloatTensor([0]), torch.cuda.FloatTensor([0])  #ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor  # clw note: 暂时不支持cpu，太慢
    bs = p[0].size(0)  # clw note: batchsize
    #print('clw: p_bs =', bs)

    obj_mask_all, noobj_mask_all, tx_all, ty_all, tw_all, th_all, tcls_all, target_all, mixup_ratios_all = build_targets(model, bs, targets)
    #print('clw: gt_bs =', target_all[0].size(0))

    # Define criteria
    #loss_reduction_type = 'sum'  # Loss reduction (sum or mean)  # reduction：控制损失输出模式。设为"sum"表示对样本进行求损失和；设为"mean"表示对样本进行求损失的平均值；而设为"none"表示对样本逐个求损失，输出与输入的shape一样。
    loss_reduction_type = 'none'
    #LabelSmooth = LabelSmoothing(reduction=loss_reduction_type)
    CELoss = nn.CrossEntropyLoss(reduction=loss_reduction_type)
    BCELossWithSigmoid = nn.BCEWithLogitsLoss(reduction=loss_reduction_type)  # withLogits的含义：输入也就是pred还会经过sigmoid, 然后再和label算二元交叉熵损失  loss = - [ ylog y^ + (1-y)log(1-y^) ]
                                                                   # 其中y^是yolo_layer层输出的结果 tcls 经过 sigmoid 函数得到的，将输出结果转换到0~1之间，即该目标属于不同类别的概率值
                                                                   # 可选参数 weight=model.class_weights ,不同类别的损失，可以设置不同的权重
    MSELoss = nn.MSELoss(reduction=loss_reduction_type)
    SmoothL1Loss = nn.SmoothL1Loss(reduction=loss_reduction_type)
    FocalLoss = MyFocalLoss(gamma=2, alpha=0.25, reduction="none")

    # Compute losses
    for i, pi in enumerate(p):  # layer index: 0,1,2   layer predictions: (bs,3,13,13,25), (bs,3,26,26,25), (bs,3,52,52,25)
        if pytorch_version_minor <= 1:   # pytorch 1.1 or less
            obj_mask = obj_mask_all[i]  # obj_mask:(64, 3, 13, 13)    obj_mask.sum: 239
            noobj_mask = noobj_mask_all[i]
        else:
            obj_mask = obj_mask_all[i].bool()   # clw note: for pytorch 1.4, have UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead
            noobj_mask = noobj_mask_all[i].bool()

        '''
        gt_boxes = target_all[i][:, :, :4] * img_size   # (64, n, 4)   n is 150 in Peter's yolov3, there is the most gt num in a batch, such as 16
        p_boxes = p_box[i]   # (64, 3, 13, 13, 4)
        p_boxes = p_boxes.float()
        p_tmp = p_boxes.unsqueeze(4)  # (64, 3, 13, 13, 1, 4)
        gt_tmp = gt_boxes.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # (64, 1, 1, 1, n, 4)
        iou = iou_xywh_torch(p_tmp, gt_tmp)  # (64, 3, 13, 13, n)
        iou_max = iou.max(-1)[0]  # clw modify:  (64, 3, 13, 13, n) ->  (64, 3, 13, 13)，即对每一个pred和n个gt比较，如果iou都小于0.5，则该pred对应位置才会计入noobj，否则不计入
        #aaa = noobj_mask.sum()  # 32, 3, 13, 13 -> sum: 16153
        noobj_mask = noobj_mask * (iou_max < 0.5)  # noobj_mask: (64, 3, 13, 13)
        #bbb = noobj_mask.sum()  # 32, 3, 13, 13 -> sum: 15868
        '''

        tx = tx_all[i][obj_mask]  # 需要把gt所在的那个grid cell的预测结果拿出来
        ty = ty_all[i][obj_mask]
        tw = tw_all[i][obj_mask]
        th = th_all[i][obj_mask]
        tcls = tcls_all[i][obj_mask]
        mixup_ratio = mixup_ratios_all[i][obj_mask]

        # Compute losses
        # clw note: 如果有 gt，也就是不是纯负样本的图，那么需要计算 （1）位置损失  （2）分类损失
        # px = torch.sigmoid(pi[obj_mask][:, 0])  # clw note：用于计算损失的是σ(tx),σ(ty),和 tw 和 th (因为gt映射到tx^时，sigmoid反函数不好求，所以不用tx和ty)
        # py = torch.sigmoid(pi[obj_mask][:, 1])
        # pw = pi[obj_mask][:, 2]
        # ph = pi[obj_mask][:, 3]
        # reduce = 'none', and then matmul with mixup_ratio; else reduce = 'sum'
        # loss_x = torch.matmul(MSELoss(px, tx), mixup_ratio) # / obj_nums  # obj_nums = len(tx)
        # loss_y = torch.matmul(MSELoss(py, ty), mixup_ratio) # / obj_nums
        # loss_w = torch.matmul(MSELoss(pw, tw), mixup_ratio) # / obj_nums
        # loss_h = torch.matmul(MSELoss(ph, th), mixup_ratio) # / obj_nums

        #### xy改为BCE，wh改为Smooth L1，  ==> mAP 0.694, ↓ ~0.007
        # px = pi[obj_mask][:, 0]
        # py = pi[obj_mask][:, 1]
        # pw = pi[obj_mask][:, 2]
        # ph = pi[obj_mask][:, 3]
        # loss_x = torch.matmul(BCELossWithSigmoid(px, tx), mixup_ratio) # / obj_nums  # obj_nums = len(tx)
        # loss_y = torch.matmul(BCELossWithSigmoid(py, ty), mixup_ratio) # / obj_nums
        # loss_w = torch.matmul(SmoothL1Loss(pw, tw), mixup_ratio) # / obj_nums
        # loss_h = torch.matmul(SmoothL1Loss(ph, th), mixup_ratio) # / obj_nums
        #### wh改为Smooth L1 ==> mAP 0.694
        # loss_w = torch.matmul(SmoothL1Loss(pw, tw), mixup_ratio) # / obj_nums
        # loss_h = torch.matmul(SmoothL1Loss(ph, th), mixup_ratio) # / obj_nums
        ####

        #### xy改为BCE  ==> mAP 0.701, P 0.338 P更高
        px = pi[obj_mask][:, 0]
        py = pi[obj_mask][:, 1]
        pw = pi[obj_mask][:, 2]
        ph = pi[obj_mask][:, 3]
        loss_x = torch.matmul(BCELossWithSigmoid(px, tx), mixup_ratio) # / obj_nums  # obj_nums = len(tx)
        loss_y = torch.matmul(BCELossWithSigmoid(py, ty), mixup_ratio) # / obj_nums
        loss_w = torch.matmul(MSELoss(pw, tw), mixup_ratio) # / obj_nums
        loss_h = torch.matmul(MSELoss(ph, th), mixup_ratio) # / obj_nums
        lbox += 2.5 * (loss_x + loss_y + loss_w + loss_h)
        #lbox += (loss_x + loss_y + loss_w + loss_h)
        ####


        ####
        # 2、计算分类损失，这里只针对多类别，如果只有1个类那么只需要计算 obj 损失
        ### (1) normal BCE
        # if model.nc > 1:  # cls loss (only if multiple classes)
        #     lcls += BCELossWithSigmoid(pi[obj_mask][:, 5:], tcls)

        ### (2) CE + label_smooth 0.1 -> mAP  0.689  P  0.178
        ###                       0.01 -> mAP 0.688 P  0.157 结论：（1）labelsmooth有提升：综合后几轮来看mAP更高更稳定，且P明显更高
        ###                                                       （2）CELoss和BCELoss相比，准确率P差的非常多，大概两倍多；
        ###     注意如果使用CE，在model.py中改为torch.sigmoid_(io[..., 4])
        # logsoftmax = nn.LogSoftmax(dim=1)
        # LabelSmooth = LabelSmoothing()
        # lcls_ = LabelSmooth( logsoftmax(pi[obj_mask][:, 5:]) , tcls.argmax(1) ).view(-1)
        # a = mixup_ratio.reshape(-1, 1)
        # b = a.repeat(1, model.nc)
        # c = b.reshape(-1)
        # lcls += torch.matmul(lcls_, c) #  / obj_nums


        # (3) BCE + wrong mixup_ratio -> mAP 0.696
        # aaa = BCELossWithSigmoid(pi[obj_mask][:, 5:], tcls).view(-1)
        # bbb = mixup_ratio.repeat(20)
        # lcls += torch.matmul(aaa, bbb) #  / obj_nums

        # (3.5) BCE + right mixup_ratio -> mAP 0.7，P=0.33 ;  (best)
        aaa = BCELossWithSigmoid(pi[obj_mask][:, 5:], tcls).view(-1)
        a = mixup_ratio.reshape(-1, 1)
        b = a.repeat(1, model.nc)
        bbb = b.reshape(-1)
        lcls += torch.matmul(aaa, bbb) #  / obj_nums

        # (3.99) BCE + right mixup_ratio + LabelSmooth factor 0.05 -> mAP 0.692  P=0.36
        # 结论：（1）BCE不适合加入labelsmooth，mAP低了0.01，并且P基本没有提升；
        # label_smooth
        # smoothing_factor = 0
        # indexes = tcls.argmax(1)
        # for i, index in enumerate(indexes):
        #     tcls[i, index] = 1 - smoothing_factor -  smoothing_factor / (model.nc - 1)   # tcls: [0,0,0,....,0,1,0,0]
        # tcls[:, :] += smoothing_factor / (model.nc - 1)  # 额外减去smoothing_factor / (model.nc - 1)，再统一加上
        # # BCELossWithSigmoid
        # aaa = BCELossWithSigmoid(pi[obj_mask][:, 5:], tcls).view(-1)
        # a = mixup_ratio.reshape(-1, 1)
        # b = a.repeat(1, model.nc)
        # bbb = b.reshape(-1)
        # lcls += torch.matmul(aaa, bbb) #  / obj_nums

        # (4) CE, inference with sigmoid  -> mAP 0.66  #  tcls是一个list，含有3个tensor，每个torch.size是308，形如 [2 1 14 14 14 6...]
        #     CE, inference without sigmoid，将models.py里面的torch.sigmoid_(io[..., 4:])改为torch.sigmoid_(io[..., 4])  -> mAP 0.681
        # lcls_ =  CELoss(pi[obj_mask][:, 5:], tcls.argmax(1))
        # lcls += torch.matmul(lcls_, mixup_ratio)


        #### 3、计算 obj 损失
        # BCELoss
        tconf = obj_mask.float()
        loss_obj = BCELossWithSigmoid(pi[obj_mask][..., 4], tconf[obj_mask])
        ###
        # loss_obj = FocalLoss(pi[obj_mask][..., 4], tconf[obj_mask])  # FocalLoss
        ###
        if loss_obj.numel():  # 有anchor iou匹配的gt
            loss_obj = torch.matmul(loss_obj, mixup_ratio)  # / obj_nums
        else:
            loss_obj = 0
        loss_noobj = BCELossWithSigmoid(pi[noobj_mask][..., 4],  tconf[noobj_mask]).sum()   #  / obj_nums
        ###
        # loss_noobj = FocalLoss(pi[noobj_mask][..., 4],  tconf[noobj_mask]).sum()   #  FocalLoss
        ###
        loss_obj_all = loss_obj + 0.5*loss_noobj
        #oss_obj_all = loss_obj + loss_noobj
        lobj += loss_obj_all


    lbox /= bs
    lobj /= bs
    lcls /= bs

    loss = lbox + lobj + lcls
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()
###


def select_device(device):  # 暂时不支持 CPU
    assert torch.cuda.is_available(), 'CUDA unavailable and CPU not support yet, invalid device: %s' % device
    if device == '':
        ng = torch.cuda.device_count()
        device = '0'
        for i in range(1, ng):
            device += ',' + str(i)
    os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable

    #nums_of_gpu = torch.cuda.device_count()
    #if nums_of_gpu > 1 and batch_size:    # TODO: 多卡，batch_size不能被卡的总数整除 check that batch_size is compatible with device_count
    #    assert batch_size % nums_of_gpu == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, nums_of_gpu)

    gpu_idxs = [int(str.strip()) for str in device.split(',')]
    x = [torch.cuda.get_device_properties(i) for i in gpu_idxs]
    s = 'Using CUDA '
    for i, gpu_idx in enumerate(gpu_idxs):
        if i == len(gpu_idxs) - 1:
            s = ' ' * len(s)
        print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" % (s, gpu_idxs[i], x[i].name, x[i].total_memory / 1024 ** 2))  # bytes to MB
    print('')

    '''
    Using CUDA device0 _CudaDeviceProperties(name='GeForce GTX 1080', total_memory=8116MB)
           device1 _CudaDeviceProperties(name='GeForce GTX 1080', total_memory=8119MB)
    '''

    return torch.device('cuda:{}'.format(gpu_idxs[0]))


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

#  clw note: 一个框和多个框算iou
def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


#  clw note: 一个框和多个框算iou
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    iou = inter_area / union_area  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union_area) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


### clw note:多个框和多个框算iou
def bboxes_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bboxes_anchor_iou(box1, anchors_wh, x1y1x2y2=True):
    anchor_xy = box1[:, :2].floor() + 0.5  # (14, 2)    中心点

    # Returns the IoU of box1 to anchors_wh.
    anchors_wh = anchors_wh  # (1,2)
    anchors_wh = anchors_wh.repeat(box1.shape[0], 1)
    anchors_xywh = torch.cat((anchor_xy, anchors_wh), 1)

    if not x1y1x2y2:  # xywh
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = anchors_xywh[:, 0] - anchors_xywh[:, 2] / 2, anchors_xywh[:, 0] + anchors_xywh[:, 2] / 2
        b2_y1, b2_y2 = anchors_xywh[:, 1] - anchors_xywh[:, 3] / 2, anchors_xywh[:, 1] + anchors_xywh[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = anchors_xywh[:, 0], anchors_xywh[:, 1], anchors_xywh[:, 2], anchors_xywh[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    # inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
    #     inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # # Union Area
    # b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    # b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    # clw modify: TODO
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def iou_xywh_torch(boxes1, boxes2):  # borrowed from Peter, xctr, yctr, w, h
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(x, y, w, h)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # 分别计算出boxes1和boxes2的左上角坐标、右下角坐标
    # 存储结构为(xmin, ymin, xmax, ymax)，其中(xmin,ymin)是bbox的左上角坐标，(xmax,ymax)是bbox的右下角坐标
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU



# modify by clw
# def non_max_suppression(prediction, conf_thres=0.5, nms_thresh=0.5):
#     """Pure Python NMS baseline."""
#
#     output = [None] * len(prediction)
#
#     for image_i, dets in enumerate(prediction):
#         if dets is None or len(dets) == 0 :
#             continue
#
#         # Multiply conf by class conf to get combined confidence
#         class_conf, class_pred = dets[:, 5:].max(1)
#         dets[:, 4] *= class_conf
#         # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
#         dets = torch.cat((dets[:, :5], class_conf.unsqueeze(1), class_pred.unsqueeze(1).float()), 1)  # clw note: (10647, 7)
#         dets = dets[dets[:, 4] > conf_thres]  # (173, 7)
#
#         # x_ctr,y_ctr,w,h -> x1,y1,x2,y2
#         dets[:, 0] -= dets[:, 2] / 2
#         dets[:, 1] -= dets[:, 3] / 2
#         dets[:, 2] += dets[:, 0]
#         dets[:, 3] += dets[:, 1]
#
#         # x1、y1、x2、y2、以及score赋值
#         x1 = dets[:, 0]     # xmin
#         y1 = dets[:, 1]     # ymin
#         x2 = dets[:, 2]      # w
#         y2 = dets[:, 3]     # h
#         scores = dets[:, 4]
#
#
#         areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#         # argsort()返回数组值从小到大的索引值
#         _, order = torch.sort(scores, descending=True)
#         order = order.cpu().numpy()
#         keep = []
#
#         while isinstance(order, np.ndarray) and len(order) > 0:  # 还有数据
#             #print(len(order))
#             i = order[0]
#             keep.append(i)
#             # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
#             xx1 = torch.max(x1[i], x1[order[1:]])
#             yy1 = torch.max(y1[i], y1[order[1:]])
#             xx2 = torch.min(x2[i], x2[order[1:]])
#             yy2 = torch.min(y2[i], y2[order[1:]])
#             # 计算相交框的面积
#             w = torch.max(torch.tensor([0.0]).cuda(), xx2 - xx1 + 1)
#             h = torch.max(torch.tensor([0.0]).cuda(), yy2 - yy1 + 1)
#             inter = w * h
#             # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
#             IOU = inter / (areas[i] + areas[order[1:]] - inter)
#             # 找到重叠度不高于阈值的矩形框索引
#             left_index = (IOU <= nms_thresh).nonzero().squeeze().cpu().numpy()
#             # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
#             order = order[left_index + 1]
#
#         det_max = dets[keep]
#         if det_max.shape != torch.Size([0]):
#             output[image_i] = det_max  # sort
#
#     return output  # list (bs, )  -> such as tensor (53, 7)


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):   # prediction: (bs, 10647, 25), 10647=(13*13+26*26+52*52)*3
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    """

    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximium box width and height
    max_boxed_per_cls = 500

    # count = 0
    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        if pred is None or len(pred) == 0 :
            continue

        # Multiply conf by class conf to get combined confidence
        class_conf, class_pred = pred[:, 5:].max(1)
        pred[:, 4] *= class_conf

        # # Merge classes (optional)
        # class_pred[(class_pred.view(-1,1) == torch.LongTensor([2, 3, 5, 6, 7]).view(1,-1)).any(1)] = 2
        #
        # # Remove classes (optional)
        # pred[class_pred != 2, 4] = 0.0

        # Select only suitable predictions
        i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & (pred[:, 2:4] < max_wh).all(1)
        pred = pred[i]

        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)


        # Get detections sorted by decreasing confidence scores
        pred = pred[(-pred[:, 4]).argsort()]

        # Set NMS method https://github.com/ultralytics/yolov3/issues/679
        # 'OR', 'AND', 'MERGE', 'VISION', 'VISION_BATCHED'
        # method = 'MERGE' if conf_thres <= 0.1 else 'VISION'  # MERGE is highest mAP, VISION is fastest
        method = 'MERGE'  # TODO

        # Batched NMS
        if method == 'VISION_BATCHED':
            i = torchvision.ops.boxes.batched_nms(boxes=pred[:, :4],
                                                  scores=pred[:, 4],
                                                  idxs=pred[:, 6],
                                                  iou_threshold=nms_thres)
            output[image_i] = pred[i]
            continue

        # Non-maximum suppression
        det_max = []

        for c in pred[:, -1].unique():  # clw note: nms by different classes
            dc = pred[pred[:, -1] == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > max_boxed_per_cls:
                dc = dc[:max_boxed_per_cls]  # limit to first 500 boxes: https://github.com/ultralytics/yolov3/issues/117

            if method == 'VISION':
                i = torchvision.ops.boxes.nms(dc[:, :4], dc[:, 4], nms_thres)
                det_max.append(dc[i])

            elif method == 'OR':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif method == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold


            elif method == 'MERGE':  # weighted mixture box

                while len(dc):
                    #start = time.time()
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes

                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]
                    # count += 1
                    #print('time use per box: %.6fs' % (time.time() - start))



            elif method == 'SOFT':  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
                    # dc = dc[dc[:, 4] > nms_thres]  # new line per https://github.com/ultralytics/yolov3/issues/362
                    dc = dc[dc[:, 4] > conf_thres]  # new line per https://github.com/ultralytics/yolov3/issues/362

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort
    # print('clw:', count)
    return output   # list (64,)  ->  (n, 7)


# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)   # 为CPU设置种子用于生成随机数，以确保结果是确定的
    #torch.cuda.manual_seed(seed) # 为当前GPU设置随机数种子
    torch.cuda.manual_seed_all(seed) # 如果使用多个GPU，应该使用为所有的GPU设置随机数种子。
                                     # Sets the seed for generating random numbers on all GPUs. It’s safe to call
                                     # this function if CUDA is not available; in that case, it is silently ignored.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # Remove randomness, slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r.append(recall[-1])

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p.append(precision[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall, precision))

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            # ax.plot(np.concatenate(([0.], recall)), np.concatenate(([0.], precision)))
            # ax.set_xlabel('YOLOv3-SPP')
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    if 'darknet53.conv.74' in weights:
        cutoff = 75
    elif 'yolov3-tiny.conv.15' in weights:
        cutoff = 15
    elif 'resnet50.pth' in weights:
        cutoff = 74
    elif 'resnet18.pth' in weights:
        cutoff = 31
    # elif 'cspdarknet53-panet-spp' in weights:  # clw note：这个 .weight文件是在coco训练的全网络的权重，加载后mAP非常高
    #     cutoff = 137

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv_layer = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            #print(i, weights[ptr])  # clw add: for debug
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

    return cutoff

def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            conf = None if gt else image_targets[:, 6]  # check for confidence presence (gt vs pred)

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic


def plot_images2(imgs, targets, paths=None, fname='images.jpg'):
    from pathlib import Path
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()
    # targets = targets[targets[:, 1] == 21]  # plot only one class

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    bs = min(bs, 16)  # limit plot to 16 images
    ns = np.ceil(bs ** 0.5)  # number of subplots

    for i in range(bs):
        boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
        plt.axis('off')
        if paths is not None:
            s = Path(paths[i]).name
            plt.title(s[:min(len(s), 40)], fontdict={'size': 8})  # limit to 40 characters
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close()


def write_to_file(text, file, mode='a'):
    with open(file, mode) as f:
        f.write(str(text) + '\n')

def print_model_biases(model):
    # prints the bias neurons preceding each yolo layer
    print('\nModel Bias Summary (per output layer):')
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for l in model.yolo_layers:  # print pretrained biases
        if multi_gpu:
            na = model.module.module_list[l].na  # number of anchors
            b = model.module.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
        else:
            na = model.module_list[l].na
            b = model.module_list[l - 1][0].bias.view(na, -1)  # bias 3x85
        print('regression: %5.2f+/-%-5.2f ' % (b[:, :4].mean(), b[:, :4].std()),
              'objectness: %5.2f+/-%-5.2f ' % (b[:, 4].mean(), b[:, 4].std()),
              'classification: %5.2f+/-%-5.2f' % (b[:, 5:].mean(), b[:, 5:].std()))


def weights_init_normal(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        #torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

        ##########3
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
        # print("initing {}".format(m))
        ###############


    elif classname.find("BatchNorm2d") != -1:

        ###############
        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)
        # print("initing {}".format(m))
        ###########

        # torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        # torch.nn.init.constant_(m.bias.data, 0.0)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape     # clw note:  resize
        ### clw modify
        coords[:, [0, 2]] =  coords[:, [0, 2]] * img0_shape[1] / img1_shape[1]
        coords[:, [1, 3]] =  coords[:, [1, 3]] * img0_shape[0] / img1_shape[0]

    else:  # clw note:  letterbox
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain

    clip_coords(coords, img0_shape)   # clw note: mAP +0.1
    return coords

def clip_coords(boxes, img_shape):

    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, np.ndarray):
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], a_min=0, a_max=img_shape[1])  # numpy:  clip x
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], a_min=0, a_max=img_shape[0])  # clip y
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=img_shape[1])  # torch:  clip x
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=img_shape[0])  #         clip y


# class LabelSmoothing(nn.Module):
#     "Implement label smoothing.  size表示类别总数  "
#     def __init__(self, smoothing=0.0, reduction='none'):  # reduction 一般是 'mean' or 'sum' or 'none'，还有个 'batchmean'
#         super(LabelSmoothing, self).__init__()
#
#         assert 0 <= smoothing < 1, 'You must have 0 <= smoothing < 1 !!'
#         self.criterion = nn.KLDivLoss(reduction=reduction) # clw note: size_average 和 reduce 这两个属性都已经Deprecated,
#         # self.padding_idx = padding_idx
#         self.confidence = 1.0 - smoothing  # if i=y的公式
#         self.smoothing = smoothing
#         self.true_dist = None
#
#     def forward(self, x, target):
#         """
#         x表示输入 (N，M)N个样本，M表示总类数，每一个类的概率log P
#         target表示label（M，）
#         """
#         self.true_dist = x.data.clone()  # 先深复制过来
#         self.true_dist.fill_(self.smoothing / (x.size(1) - 1))  # otherwise的公式
#         self.true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         #  clw note: “1” 表示按列填充， 在相应位置处填入self.confidence这个值
#         #            target.data即相应class的索引，unsqueeze(1)后shape为(n, 1)
#
#         # clw note: not good ?
#         # loss = self.criterion(x.log(), self.true_dist)
#         # loss = loss.sum(axis=1, keepdim=False)
#         # loss = loss.mean()
#         # return loss
#
#         loss = -torch.log(x) * self.true_dist
#         # loss = loss.sum(dim=1)
#         # loss = loss.mean()
#         return loss


class LabelSmoothing(nn.NLLLoss):
    """A soft version of negative log likelihood loss with support for label smoothing.

    Effectively equivalent to PyTorch's :class:`torch.nn.NLLLoss`, if `label_smoothing`
    set to zero. While the numerical loss values will be different compared to
    :class:`torch.nn.NLLLoss`, this loss results in the same gradients. This is because
    the implementation uses :class:`torch.nn.KLDivLoss` to support multi-class label
    smoothing.

    Args:
        label_smoothing (float):
            The smoothing parameter :math:`epsilon` for label smoothing. For details on
            label smoothing refer `this paper <https://arxiv.org/abs/1512.00567v1>`__.
        weight (:class:`torch.Tensor`):
            A 1D tensor of size equal to the number of classes. Specifies the manual
            weight rescaling applied to each class. Useful in cases when there is severe
            class imbalance in the training set.
        num_classes (int):
            The number of classes.
        size_average (bool):
            By default, the losses are averaged for each minibatch over observations **as
            well as** over dimensions. However, if ``False`` the losses are instead
            summed. This is a keyword only parameter.
    """

    def __init__(self, label_smoothing=0.1, num_classes=20, **kwargs):
        super(LabelSmoothing, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.confidence = 1 - self.label_smoothing
        self.num_classes = num_classes

        assert label_smoothing >= 0.0 and label_smoothing <= 1.0

        self.criterion = nn.KLDivLoss(reduction='none', **kwargs)

    def forward(self, input, target):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.label_smoothing / (self.num_classes - 1))
        one_hot.scatter_(1, target.unsqueeze(1).long(), self.confidence)

        return self.criterion(input, one_hot)
