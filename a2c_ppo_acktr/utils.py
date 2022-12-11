import shutil
import torch
import torch.nn as nn
import random 
import json
import numpy as np
import random
import os
import pickle
import time

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)


def load_json(path):
    with open(path,'r') as f:
        res = json.load(f)
    return res


def save_json(obj, path:str):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=4)


def load_pkl(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
        return res


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def setup_seed(seed = 3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # https://zhuanlan.zhihu.com/p/73711222
    # torch.backends.cudnn.benchmark = True


def get_datetime():
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    return t


class Logger():
    def __init__(self,log_file_path) -> None:
        self.path = log_file_path
        with open(self.path,'w') as f:
            f.write(get_datetime() + "\n")
            print(get_datetime())
        return
    
    def log(self,content):
        content = str(content)
        with open(self.path,'a') as f:
            f.write(content + "\n")
            print(content)
        return


def mkdir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

