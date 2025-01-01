import torch
import pickle
from os.path import join
from resnet import resnet50_features, resnet101_features
from model_mg_rfm_mr_scam import AttentionNet

import os
import numpy as np
def set_seed(seed):
#   import random
#   random.seed(seed)
#   print('setting random-seed to {}'.format(seed))

#   import numpy as np
#   np.random.seed(seed)
#   print('setting np-random-seed to {}'.format(seed))

#   import torch
#   print('cudnn.enabled set to {}'.format(torch.backends.cudnn.enabled))
#   # set seed for CPU
#   torch.manual_seed(seed)
#   torch.cuda.manual_seed_all(seed)
#   print('setting torch-seed to {}'.format(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def make_optimizer(cfg, model):
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    momentum = cfg.SOLVER.MOMENTUM

    params_to_update = []
    params_names = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            params_names.append(name)

    optimizer = torch.optim.SGD(params_to_update, lr=lr, weight_decay=weight_decay, momentum=momentum)
    return optimizer

def make_lr_scheduler(cfg, optimizer, n_iter_per_epoch):
    step_size = cfg.SOLVER.STEPS
    gamma = cfg.SOLVER.GAMMA
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def load_model(cfg):
    dataset_name = cfg.DATASETS.NAME
    att_type = cfg.DATASETS.SEMANTIC_TYPE
    info = get_attributes_info(dataset_name, att_type)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    scls_num = cls_num-ucls_num

    attr_group = get_attr_group(dataset_name)

    img_size = cfg.DATASETS.IMAGE_SIZE
    dataset_name = cfg.DATASETS.NAME
    
    hid_dim = cfg.MODEL.HID_DIM
    scale = cfg.MODEL.SCALE
    pretrained = cfg.MODEL.BACKBONE.PRETRAINED
    ft_flag = cfg.MODEL.BACKBONE.FINETUNE
    model_dir = cfg.PRETRAINED_MODELS

    c, w, h = 2048, img_size // 32, img_size // 32
    backbone = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    
    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device(cfg.MODEL.DEVICE)
    
    return AttentionNet(backbone=backbone, ft_flag = ft_flag, img_size=img_size,
                  hid_dim=hid_dim, c=c, w=w, h=h, scale=scale,
                  attritube_num=attritube_num,
                  attr_group=attr_group, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num, device=device)

def get_attributes_info(name, att_type):
    if "CUB" in name:
        if att_type == "GBU":
            info = {
                "input_dim" : 312,
                "n" : 200,
                "m" : 50
            }
        elif att_type == "VGSE":
            info = {
                "input_dim": 469,
                "n": 200,
                "m": 50
            }
    elif "AWA" in name or "AwA" in name:
        if att_type=="GBU":
            info = {
                "input_dim": 85,
                "n": 50,
                "m": 10
            }
        elif att_type=="VGSE":
            info = {
                "input_dim": 250,
                "n": 50,
                "m": 10
            }
    elif "SUN" in name:
        if att_type == "GBU":
            info = {
                "input_dim": 102,
                "n": 717,
                "m": 72
            }
        elif att_type == "VGSE":
            info = {
                "input_dim": 450,
                "n": 717,
                "m": 72
            }
    else:
        info = {}
    return info

def get_attr_group(name):
    if "CUB" in name:
        attr_group = {
            1: [i for i in range(0, 9)],
            2: [i for i in range(9, 24)],
            3: [i for i in range(24, 39)],
            4: [i for i in range(39, 54)],
            5: [i for i in range(54, 58)],
            6: [i for i in range(58, 73)],
            7: [i for i in range(73, 79)],
            8: [i for i in range(79, 94)],
            9: [i for i in range(94, 105)],
            10: [i for i in range(105, 120)],
            11: [i for i in range(120, 135)],
            12: [i for i in range(135, 149)],
            13: [i for i in range(149, 152)],
            14: [i for i in range(152, 167)],
            15: [i for i in range(167, 182)],
            16: [i for i in range(182, 197)],
            17: [i for i in range(197, 212)],
            18: [i for i in range(212, 217)],
            19: [i for i in range(217, 222)],
            20: [i for i in range(222, 236)],
            21: [i for i in range(236, 240)],
            22: [i for i in range(240, 244)],
            23: [i for i in range(244, 248)],
            24: [i for i in range(248, 263)],
            25: [i for i in range(263, 278)],
            26: [i for i in range(278, 293)],
            27: [i for i in range(293, 308)],
            28: [i for i in range(308, 312)],
        }

    elif "AWA" in name or "AwA" in name:
        attr_group = {
            1: [i for i in range(0, 8)],
            2: [i for i in range(8, 14)],
            3: [i for i in range(14, 18)],
            4: [i for i in range(18, 34)]+[44, 45],
            5: [i for i in range(34, 44)],
            6: [i for i in range(46, 51)],
            7: [i for i in range(51, 63)],
            8: [i for i in range(63, 78)],
            9: [i for i in range(78, 85)],
        }

    elif "SUN" in name:
        attr_group = {
            1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
            2: [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]+[80, 99],
            3: [74, 75, 76, 77, 78, 79, 81, 82, 83, 84, ],
            4: [85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98] + [100, 101]
        }
    else:
        attr_group = {}

    return attr_group
