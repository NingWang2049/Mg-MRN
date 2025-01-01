import os
from os.path import join
import sys
import argparse

import torch
import numpy as np
from data import build_dataloader

from utils.config import cfg
from utils.utils import *
from utils.regLoss import weighted_RegressLoss
from inferencer import eval_zs_gzsl

def train_model(cfg):
    device = cfg.MODEL.DEVICE
    model = load_model(cfg).to(device)
    tr_dataloader, tu_loader, ts_loader, res = build_dataloader(cfg)
    
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer, len(tr_dataloader))
    
    output_dir = cfg.OUTPUT_DIR
    model_file_name = cfg.MODEL_FILE_NAME
    model_file_path = join(output_dir, model_file_name)

    test_gamma = cfg.TEST.GAMMA
    max_epoch = cfg.SOLVER.MAX_EPOCH

    RegNorm = cfg.MODEL.LOSS.REG_NORM
    RegType = cfg.MODEL.LOSS.REG_TYPE
    scale = cfg.MODEL.SCALE

    info = get_attributes_info(cfg.DATASETS.NAME, cfg.DATASETS.SEMANTIC_TYPE)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    scls_num = cls_num - ucls_num

    lamd = {
        0: cfg.MODEL.LOSS.LAMBDA0,
        1: cfg.MODEL.LOSS.LAMBDA1,
        2: cfg.MODEL.LOSS.LAMBDA2,
        3: cfg.MODEL.LOSS.LAMBDA3,
    }

    best_performance = [-0.1, -0.1, -0.1, -0.1, -0.1] # ZSL, S, U, H, AUSUC
    best_epoch = -1
    att_all = res['att_all'].to(device)
    att_all_var = torch.var(att_all,dim=0)
    att_all_std = torch.sqrt(att_all_var+1e-12)
    print(att_all_std)
    att_seen = res['att_seen'].to(device)
    support_att_seen=att_seen

    print("-----use "+ RegType + " -----")
    Reg_loss = weighted_RegressLoss(RegNorm, RegType, device)
    CLS_loss = torch.nn.CrossEntropyLoss()

    losses = []
    cls_losses = []
    reg_losses = []

    model.train()

    for epoch in range(0, max_epoch):
        print("lr: %.8f"%(optimizer.param_groups[0]["lr"]))

        loss_epoch = []
        cls_loss_epoch = []
        reg_loss_epoch = []

        scheduler.step()

        num_steps = len(tr_dataloader)
        for iteration, (batch_img, batch_att, batch_label) in enumerate(tr_dataloader):
            batch_img = batch_img.to(device)
            batch_att = batch_att.to(device)
            batch_label = batch_label.to(device)
            
            v2s1, _, _ = model(x=batch_img, support_att=support_att_seen)
            score1, cos_dist = model.cosine_dis(pred_att=v2s1, support_att=support_att_seen, stage='1')
            Lreg = Reg_loss(v2s1, batch_att, weights = None)
            Lcls = CLS_loss(score1, batch_label)
            loss1 = lamd[0] * Lcls + lamd[1] * Lreg
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            
            _, v2s2, _ = model(x=batch_img, support_att=support_att_seen)
            score2, cos_dist = model.cosine_dis(pred_att=v2s2, support_att=support_att_seen, stage='1')
            Lreg = Reg_loss(v2s2, batch_att, weights = None)
            Lcls = CLS_loss(score2, batch_label)
            loss2 = lamd[0] * Lcls + lamd[1] * Lreg
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            
            _, _, v2s3 = model(x=batch_img, support_att=support_att_seen)
            score3, cos_dist = model.cosine_dis(pred_att=v2s3, support_att=support_att_seen, stage='1')
            Lreg = Reg_loss(v2s3, batch_att, weights = None)
            Lcls = CLS_loss(score3, batch_label)
            loss3 = lamd[0] * Lcls + lamd[1] * Lreg
            optimizer.zero_grad()
            loss3.backward()
            optimizer.step()
            loss = loss1 + loss2 + loss3
            
            log_info = 'epoch: %d, it: %d/%d  |  loss: %.4f, cls_loss: %.4f, reg_loss: %.4f, lr: %.10f' % \
                        (epoch + 1, iteration, num_steps, loss, Lcls, Lreg, optimizer.param_groups[0]["lr"])
            print(log_info)
            
            loss_epoch.append(loss.item())
            cls_loss_epoch.append(Lcls.item())
            reg_loss_epoch.append(Lreg.item())

        losses += loss_epoch
        cls_losses += cls_loss_epoch
        reg_losses += reg_loss_epoch

        loss_epoch_mean = sum(loss_epoch)/len(loss_epoch)
        cls_loss_epoch_mean = sum(cls_loss_epoch)/len(cls_loss_epoch)
        reg_loss_epoch_mean = sum(reg_loss_epoch)/len(reg_loss_epoch)
        log_info = 'epoch: %d |  loss: %.4f, cls_loss: %.4f, reg_loss: %.4f, lr: %.10f' % \
                    (epoch + 1, loss_epoch_mean, cls_loss_epoch_mean, reg_loss_epoch_mean, optimizer.param_groups[0]["lr"])
        print(log_info)
        
        acc_seen, acc_novel, H, acc_zs, AUSUC, best_gamma = eval_zs_gzsl(
            tu_loader,
            ts_loader,
            res,
            model,
            test_gamma,
            device)
        
        print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f, AUSUC=%.4f, best_gamma=%.4f' % (acc_zs, acc_seen, acc_novel, H, AUSUC, best_gamma))

        if acc_zs > best_performance[0]:
            best_performance[0] = acc_zs

        if H > best_performance[3]:
            best_epoch=epoch+1
            best_performance[1:4] = [acc_seen, acc_novel, H]
            data = {}
            data["model"] = model.state_dict()
            torch.save(data, model_file_path)
            print('save best model: ' + model_file_path)
        
        if AUSUC > best_performance[4]:
            best_performance[4] = AUSUC
            model_file_path_AUSUC = model_file_path.split('.pth')[0]+'_AUSUC'+'.pth'
            torch.save(data, model_file_path_AUSUC)
            print('save best AUSUC model: ' + model_file_path_AUSUC)
        print("best: ep: %d" % best_epoch)
        print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f, AUSUC=%.4f' % tuple(best_performance))

    print("best: ep: %d" % best_epoch)
    print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f, AUSUC=%.4f' % tuple(best_performance))

def main():
    parser = argparse.ArgumentParser(description="PyTorch Zero-Shot Learning Training")
    parser.add_argument(
        "--config-file",
        default="config/cub_16w_2s_AttentionNet.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()
    
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))
    
    seed = cfg.SEED
    set_seed(seed)

    train_model(cfg)

if __name__ == '__main__':
    main()
