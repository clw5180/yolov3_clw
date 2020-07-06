from torch.utils.data import Dataset
import torch
import numpy as np
import os
import cv2
from .transforms import build_transforms
import random
import math

from utils.utils import write_to_file
from utils.globals import log_file_path

'''
读取自己数据的基本流程：
1：制作存储了图像的路径和标签信息的txt9
2：将这些信息转化为list，该list每一个元素对应一个样本
3：通过getitem函数，读取数据标签，并返回。

可以把voc2007数据集拷过来，测一下
'''

class VocDataset(Dataset):  # for training/testing
    '''
    类功能：给定训练集或测试集txt所在路径(该txt包含了训练集每一张图片的路径，一行对应一张图片，如 home/dataset/voc2007/train/cat1.jpg),
    以及图片大小img_size，制作可用于迭代的训练集；
    适用目录结构：cat1.txt放置在和cat1.jpg同一文件夹下，cat1.txt是由当前目录下的cat1.xml通过 xml2txt.py脚本转化而来
    '''
    def __init__(self, txt_path, img_size, with_label, is_training):  # clw note: (1) with_label=True, is_training=True -> train
                                                                      #           (2) with_label=True, is_training=False -> test(no aug)
                                                                      #           (3) with_label=False, is_training=False -> detect
        # 1、获取所有图片路径，存入 list
        with open(txt_path, 'r') as f:
            self.img_file_paths = [x.replace(os.sep, '/') for x in f.read().splitlines()]
        assert len(self.img_file_paths) > 0, 'No images found in %s !' % txt_path

        # 2、获取所有 txt 路径，存入 list
        self.label_file_paths = []
        for img_file_path in self.img_file_paths:
            txt_file_path = img_file_path[:-4] + '.txt'
            assert os.path.isfile(txt_file_path), 'No label_file %s found, maybe need to exec xml2txt.py first !' % txt_file_path
            self.label_file_paths.append(txt_file_path)   # 注意除了有 .jpg .png可能还有.JPG甚至其他...
        if len(self.label_file_paths) == 0:
            with_label = False

        self.mosaic = False
        if is_training and with_label:
            self.mosaic = False  # clw note: TODO: if use mosaic, remove RandomCrop(), RandomAffine(), LetterBox(img_size), in transforms.py
            if self.mosaic:
                print('using mosaic !')
                write_to_file('using mosaic !', log_file_path)

        # 3、transforms and data aug，如必须要做的 Resize(), ToTensor()
        self.transforms = build_transforms(img_size, is_training)
        self.img_size = img_size
        self.with_label = with_label
        self.training = is_training


    def __len__(self):
        return len(self.img_file_paths)

    def __getitem__(self, index):
        # 1、根据 index 读取相应图片，保存图片信息；如果是train/test 还需要读入label
        if self.with_label:  # train or test

            if self.mosaic:
                img, label, img_path = load_mosaic(self, index)
            else:
                img, label, img_path = load_image(self, index)  # labels: ndarray (n, 5)
            labels = torch.zeros((len(label), 6))  # add one column to save batch_idx   # now labels: [ batch_idx, class_idx, x, y, w, h ]
            labels[:, 1:] = torch.from_numpy(label)  # batch_idx is at the first colume, index 0

            if self.training:  # train
                img_tensor, label_tensor = self.transforms(img, labels)  # 对 img 和 label 都要做相应的变换
                return img_tensor, label_tensor, img_path

            else:  # test
                img_tensor, label_tensor, shape = self.transforms(img, labels)   # clw note: shape need to convert pred coord -> orig coord, then compute mAP
                return img_tensor, label_tensor, img_path, shape                 #           don't support RandomCrop, RandomAffline... for test, because of the coord convert is not easy

        else:   # detect
            img, img_path = load_image(self, index)
            img_tensor, shape = self.transforms(img)
            return img_tensor, img_path, shape


    @staticmethod
    def train_collate_fn(batch):
        img, label, path = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path  # TODO：如 batch=4，需要对img进行堆叠
        # img 堆叠后变成[bs, 3, 416, 416] 多了bs一个维度,   label原本是[5, 5]  [1, 5]，concat后变成 [n, 5]

    @staticmethod
    def test_collate_fn(batch):
        img, label, path, shapes = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes  # TODO：如 batch=4，需要对img进行堆叠
        # img 堆叠后变成[bs, 3, 416, 416] 多了bs一个维度,   label原本是[5, 5]  [1, 5]，concat后变成 [n, 5]

    @staticmethod
    def detect_collate_fn(batch):
        img_tensor, img_path, shape = list(zip(*batch))  # transposed
        img_tensor = torch.stack(img_tensor, 0)
        return img_tensor, img_path, shape  # TODO：如 batch=4，需要对img和label进行堆叠

def load_image(self, index):
    img_path = self.img_file_paths[index]
    img = cv2.imread(img_path)
    if img is None:
        raise Exception('Read image error: %s not exist !' % img_path)
    if self.with_label:
        label_path = self.label_file_paths[index]
        with open(label_path, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            if len(labels) == 0:
                raise Exception('Not support pure negative sample yet!')
        return img, labels, img_path
    else:
        return img, img_path


def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    img4 = np.zeros((s * 2, s * 2, 3), dtype=np.uint8) + 128  # base image with 4 tiles
    indices = [index] + [random.randint(0, len(self.img_file_paths) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):

        # Load image
        img, x, img_path = load_image(self, index)
        ######  clw add: for debug
        # if '2008_008470' in img_path:
        #     print('bbb')
        # if '2008_002738' in img_path:
        #     print('bbb')
        # if '2008_000742' in img_path:
        #    print('bbb')
        # if 5211 == index and i == 0:
        #    print('bbb')
        ######


        h, w, _ = img.shape

        # place img in img4
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
        #  x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
        #  x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
        #  x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
        #  x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        x1b = int(random.uniform(0, w - (x2a - x1a)))
        y1b = int(random.uniform(0, h - (y2a - y1a)))
        padw = x1a - x1b
        padh = y1a - y1b
        x2b = x2a - padw
        y2b = y2a - padh

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # crop at origin image a, and mapping it to mosaic image b

        # Load labels
        if x.size > 0:
            # coord mapping from origin img to mosaic img by adding pad, and convert Normalized xywh to pixel xyxy format
            labels = x.copy()
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh

            mask = (labels[:, 3] > x1a) & (labels[:, 1] < x2a) & (labels[:, 4] > y1a) & (labels[:, 2] < y2a)  # filter the gt which is out of the cropped region of origin image
            #mask = (labels[:, 1] > x1a) & (labels[:, 3] < x2a) & (labels[:, 2] > y1a) & (labels[:, 4] < y2a)
            labels = labels[mask]
            if labels.size == 0:
                labels = np.zeros((0, 5), dtype=np.float32)
                labels4.append(labels)
                continue
            # np.clip(labels[:, 1::2], x1b + padw, x2b + padw, out=labels[:, 1::2])
            # np.clip(labels[:, 2::2], y1b + padh, y2b + padh, out=labels[:, 2::2])
            np.clip(labels[:, 1::2], x1a, x2a, out=labels[:, 1::2])  # clw modify: the same with up
            np.clip(labels[:, 2::2], y1a, y2a, out=labels[:, 2::2])
            mask2 = (labels[:, 3] - labels[:, 1] > 5) & (labels[:, 4] - labels[:, 2] > 5) # filter the cropped gt which is too thin, which may lead to 'RuntimeError: CUDA error: device-side assert triggered' when build_target() if using hflip data_augment(index 13-0, 26-0, 52-0 cause this problem)
            labels = labels[mask2]
        else:
            labels = np.zeros((0, 5), dtype=np.float32)
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    if len(labels4) == 0:  #
        print('Warning: 4 img have 0 gt !!!')
        print('Warning: 4 img have 0 gt !!!')
        print('Warning: 4 img have 0 gt !!!')

    # for example: if border=-416/2=-208, it means remain img size is 832-2*208=416 -> choose a 416*416 from 832x832
    img, labels = random_affine(img4, labels4,
                                  degrees=3.07,
                                  translate=0.04,
                                  scale=0.06,
                                  shear=0.28,
                                  border=-s // 2)  # border to remove
    ###### clw add: save the image, for debug
    # img_out = img.copy()
    # for box in labels:
    #     img_out = cv2.rectangle(img, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), color=(0, 0, 255), thickness=2)  #  # TODO: 如果 box 是tensor 格式，就不需要转int   TODO
    # i = random.randint(1, 100)
    # cv2.imwrite('./mosaic_debug{}.jpg'.format(i), img_out)
    ###

    ### clw modify:
    box_xctr = (labels[:, 1] + labels[:, 3])/2 / s
    box_yctr = (labels[:, 2] + labels[:, 4])/2 / s
    box_w = (labels[:, 3] - labels[:, 1]) / s
    box_h = (labels[:, 4] - labels[:, 2]) / s
    ###

    labels[:, 1] = box_xctr
    labels[:, 2] = box_yctr
    labels[:, 3] = box_w
    labels[:, 4] = box_h

    return img, labels, img_path


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:  # targets = [cls, xyxy]
        targets = []

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    changed = (border != 0) or (M != np.eye(3)).any()
    if changed:
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_AREA, borderValue=(128, 128, 128))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets