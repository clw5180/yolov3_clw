import random
import numpy as np
import random
import cv2
from utils.globals import log_file_path
from utils.utils import write_to_file


def build_transforms(img_size, is_train=False):
    if is_train:
        transform = Compose([
            RandomHorizontalFlip(),
            RandomCrop(),
            RandomTranslate(),
            ImgProcess(img_size)  # pad not support yet
            # Resize(img_size),
            # Normalize(),
        ])

        ###### TODO:
        # transform = Compose([
        #     RandomColorDistort(),
        #     RandomExpand(),
        #     RandomCropWithConstraints(),
        #     Resize(img_size),
        #     RandomFlip(),
        #     Normalize(),
        # ])
    else:
        transform = Compose([
            # Resize(img_size),
            # Normalize(),
            ImgProcess(img_size)
        ])

        # ###### TODO:
        # transform = Compose([
        #     Resize(img_size),
        #     Normalize(),
        # ])
    return transform


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, *data):  # 测试集只用传img，训练集需要同时传入img和label；因此 data 是可变长参数
        # 这里data一定是tuple，根据长度判断data的有效性
        if len(data) > 2:
            raise Exception('can not pass more than 2 params!')
        elif len(data) == 1:
            data = data[0]   # 如果是tuple内只含有1个元素，则解除tuple，便于后面迭代 data = t(data) 的时候统一返回 img 即可

        for t in self.transforms:
            data = t(data)
        return data

    def add(self, transform):
        self.transforms.append(transform)



class RandomTranslate(object):
    def __init__(self, p=0.5):
        self.p = p
        print('using RandomTranslate !')
        write_to_file('using RandomTranslate !', log_file_path)

    def __call__(self, data):
        if not isinstance(data, tuple):
            raise Exception('Warning: RandomTranslate only support training process now !!')
        else:
            if random.random() < self.p:
                img, bboxes = data[0], data[1]
                h_img, w_img, _ = img.shape
                # 得到可以包含所有bbox的最大bbox
                max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
                max_l_trans = max_bbox[0]
                max_u_trans = max_bbox[1]
                max_r_trans = w_img - max_bbox[2]
                max_d_trans = h_img - max_bbox[3]

                tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
                ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

                M = np.array([[1, 0, tx], [0, 1, ty]])
                img = cv2.warpAffine(img, M, (w_img, h_img))

                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
                return (img, bboxes)
            else:
                return data


class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p
        print('using RandomCrop !')
        write_to_file('using RandomCrop !', log_file_path)

    def __call__(self, data):
        if not isinstance(data, tuple):
            raise Exception('Warning: RandomCrop only support training process now !!')
        else:
            if random.random() < self.p:
                img, bboxes = data[0], data[1]
                h_img, w_img, _ = img.shape
                # 得到可以包含所有bbox的最大bbox
                max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
                max_l_trans = max_bbox[0]
                max_u_trans = max_bbox[1]
                max_r_trans = w_img - max_bbox[2]
                max_d_trans = h_img - max_bbox[3]

                crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
                crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
                crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
                crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

                img = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
                return (img, bboxes)
            else:
                return data


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        print('using RandomHorizontalFlip !')
        write_to_file('using RandomHorizontalFlip !', log_file_path)

    def __call__(self, data):
        if not isinstance(data, tuple):
            raise Exception('Warning: RandomHorizontalFlip only support training process now !!')
        else:
            if random.random() < self.p:
                img, bboxes = data[0], data[1]
                _, w_img, _ = img.shape
                img = img[:, ::-1, :]
                bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
                return (img, bboxes)
            else:
                return data


class ImgProcess(object):
    def __init__(self, target_shape):
        self.target_shape = target_shape
        self.already_showed_sample = True
        print('using ImgProcess !')
        write_to_file('using ImgProcess !', log_file_path)

    def __call__(self, data):
        if not isinstance(data, tuple):
            raise Exception('Warning: ImgProcess only support training process now !!')
        else:
            """
            RGB转换 -> resize(resize不改变原图的高宽比) -> normalize
            并可以选择是否校正bbox
            :param image_org: 要处理的图像
            :param target_shape: 对图像处理后，期望得到的图像shape，存储格式为(h, w)
            :return: 处理之后的图像，shape为target_shape
            """
            image, bboxes = data[0], data[1]
            if isinstance(self.target_shape, tuple):
                h_target, w_target = self.target_shape
            else:
                h_target, w_target = self.target_shape, self.target_shape
            h_org, w_org, _ = image.shape

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            resize_ratio = min(1.0 * w_target / w_org, 1.0 * h_target / h_org)
            resize_w = int(resize_ratio * w_org)
            resize_h = int(resize_ratio * h_org)
            image_resized = cv2.resize(image, (resize_w, resize_h))
            image_paded = np.full((h_target, w_target, 3), 128.0)
            dw = int((w_target - resize_w) / 2)
            dh = int((h_target - resize_h) / 2)
            image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh

            ######
            if not self.already_showed_sample:
                img_out = image_paded.copy()
                for box in bboxes:
                    xmin = int(box[0])
                    ymin = (box[1])
                    xmax = (box[2])
                    ymax = (box[3])
                    img_out = cv2.rectangle(img_out, (xmin, ymin), (xmax, ymax), color=(0, 0, 255))
                cv2.imwrite('crop.jpg', img_out)
                self.already_showed_sample = True
            ######

            image = image_paded / 255.0
            return (image, bboxes, ((h_org, w_org), ((resize_ratio,resize_ratio), (dw, dh))))



#############################################################
# clw note: not used yet
#############################################################
import dataset.transforms.image as timage
import dataset.transforms.bbox as tbbox

class RandomColorDistort(object):
    def __init__(self):
        pass
    def __call__(self, data):
        if not isinstance(data, tuple):
            raise Exception('Warning: RandomColorDistort only support training process now !!')
        else:
            img, bbox = data[0], data[1]
            img = timage.random_color_distort(img)
            return (img, bbox)


class RandomExpand(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        if not isinstance(data, tuple):
            raise Exception('Warning: RandomExpand only support training process now !!')
        else:
            if random.random() < self.p:
                img, bbox = data[0], data[1]
                img, expand = timage.random_expand(img)
                bbox = tbbox.translate(bbox, x_offset=expand[0], y_offset=expand[1])
                return (img, bbox)
            else:
                return data


class RandomCropWithConstraints(object):
    def __init__(self):
        pass
    def __call__(self, data):
        if not isinstance(data, tuple):
            raise Exception('Warning: RandomCropWithConstraints only support training process now !!')
        else:
            img, bbox = data[0], data[1]
            h, w, _ = img.shape
            bbox, crop = tbbox.random_crop_with_constraints(bbox, (w, h))
            x0, y0, w, h = crop
            img = timage.fixed_crop(img, x0, y0, w, h)
            return (img, bbox)


class Resize(object):
    def __init__(self, new_size):
        self.new_size = new_size
    def __call__(self, data):
        if not isinstance(data, tuple):
            return timage.img_resize(data, out_size=(self.new_size, self.new_size))
        else:
            img, bbox = data[0], data[1]
            h, w, _ = img.shape
            img = timage.img_resize(img, out_size=(self.new_size, self.new_size))
            bbox = tbbox.bbox_resize(bbox, (w, h), (self.new_size, self.new_size))
            return (img, bbox)


class RandomFlip(object):
    def __init__(self):
        pass
    def __call__(self, data):
        if not isinstance(data, tuple):
            raise Exception('Warning: RandomFlip only support training process now !!')
        else:
            img, bbox = data[0], data[1]
            h, w, _ = img.shape
            img, flips = timage.random_flip(img, px=1)
            bbox = tbbox.bbox_flip(bbox, (w, h), flip_x=flips[0])
            return (img, bbox)


class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std

    def __call__(self, data):
        if not isinstance(data, tuple):
            return timage.imnormalize(data, self._mean, self._std)
        else:
            img, bbox = data[0], data[1]
            img = timage.imnormalize(img, self._mean, self._std)
            return (img, bbox)

