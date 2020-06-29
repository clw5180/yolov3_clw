import numpy as np


class CosineDecayLR(object):
    def __init__(self, optimizer, T_max, lr_init, lr_min=0., warmup=0):
        """
        a cosine decay scheduler about steps, not epochs.
        :param optimizer: ex. optim.SGD
        :param T_max:  max steps, and steps=epochs * batches
        :param lr_max: lr_max is init lr.
        :param warmup: in the training begin, the lr is smoothly increase from 0 to lr_init, which means "warmup",
                        this means warmup steps, if 0 that means don't use lr warmup.
        """
        super(CosineDecayLR, self).__init__()
        self.__optimizer = optimizer
        self.__T_max = T_max
        self.__lr_min = lr_min
        self.__lr_max = lr_init
        self.__warmup = warmup


    def step(self, t):
        if self.__warmup and t < self.__warmup:
            lr = self.__lr_max / self.__warmup * t
        else:
            T_max = self.__T_max - self.__warmup
            t = t - self.__warmup
            lr = self.__lr_min + 0.5 * (self.__lr_max - self.__lr_min) * (1 + np.cos(t/T_max * np.pi))
        for param_group in self.__optimizer.param_groups:
            param_group["lr"] = lr



def adjust_learning_rate(optimizer, gamma, lr0, total_epochs, epoch, iteration, epoch_size):
    """调整学习率进行warm up和学习率衰减
    """
    step_index = 0
    if epoch < 6:
        # 对开始的6个epoch进行warm up
        lr = 1e-6 + (lr0 - 1e-6) * iteration / (epoch_size * 2)
    else:
        if epoch > total_epochs * 0.7:
            # 在进行总epochs的70%时，进行以gamma的学习率衰减
            step_index = 1
        if epoch > total_epochs * 0.9:
            # 在进行总epochs的90%时，进行以gamma^2的学习率衰减
            step_index = 2

        lr = lr0 * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr