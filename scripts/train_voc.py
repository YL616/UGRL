import argparse
import datetime
import logging
import os
import random
import sys
sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from datasets import voc
from utils.losses import get_seg_loss, VICRegLoss
from utils import evaluate, imutils
from utils.AverageMeter import AverageMeter
from utils.camutils import (cam_to_label, multi_scale_cam, ignore_img_box)
from utils.optimizer import PolyWarmupAdamW
from utils.uncertainty import UncertaintyCalculator
from ugrl.model_ugrl import UGRL


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")

parser.add_argument("--high_thre", default=0.55, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.35, type=float, help="low_bkg_score")

parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def get_down_size(ori_shape=(512,512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w

def validate(model=None, data_loader=None, cfg=None):
    preds, gts, cams = [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            model_ddp = model.module if hasattr(model, 'module') else model

            cls, segs, _, _ = model_ddp(inputs,)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            
            _cams = multi_scale_cam(model_ddp, inputs, cfg.cam.scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, cfg=cfg)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            valid_label = torch.nonzero(cls_label[0])[:,0]
            out_cam = torch.squeeze(resized_cam)[valid_label]

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    model.train()
    return cls_score, seg_score, cam_score

def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    return (bg_loss + fg_loss) * 0.5

def train(cfg):
    num_workers = 10
    
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend,)
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    train_dataset = voc.VOC12ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    
    train_sampler = DistributedSampler(train_dataset,shuffle=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              sampler=train_sampler,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)

    ugrl = UGRL(backbone=cfg.backbone.config,
                stride=cfg.backbone.stride,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,
                pretrained=True,
                pooling=args.pooling,
                vicreg_cfg=cfg.vicreg)
    if dist.get_rank() == 0:
        logging.info('\nNetwork config: \n%s'%(ugrl))
    param_groups = ugrl.get_param_groups()
    
    ugrl.to(args.local_rank)
    
    in_channels_for_uc = ugrl.in_channels[-1]
    num_fg_classes = cfg.dataset.num_classes - 1
    uncertainty_calculator = UncertaintyCalculator(
        in_channels=in_channels_for_uc, 
        num_classes=num_fg_classes,
        d=cfg.uncertainty.hyper_dim
    )
    
    uncertainty_calculator.to(args.local_rank)
    
    vicreg_criterion = VICRegLoss(
        lmbda=cfg.vicreg.lmbda,
        mu=cfg.vicreg.mu,
        nu=cfg.vicreg.nu,
        gamma=cfg.vicreg.gamma,
        eps=cfg.vicreg.eps
    )
    vicreg_criterion.to(args.local_rank)
    if dist.get_rank() == 0:
        logging.info('\nVICReg Loss is enabled.')
        
    if dist.get_rank() == 0:
        writer = SummaryWriter(cfg.work_dir.tb_logger_dir)
    
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[4],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    if dist.get_rank() == 0:
        logging.info('\nOptimizer: \n%s' % optimizer)
    
    ugrl = DistributedDataParallel(ugrl, device_ids=[args.local_rank], find_unused_parameters=True)

    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()
    bkg_cls = torch.ones(size=(cfg.train.samples_per_gpu, 1))

    for n_iter in range(cfg.train.max_iters):
        try:
            img_name, inputs1, inputs2, cls_labels, img_box1, img_box2 = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs1, inputs2, cls_labels, img_box1, img_box2 = next(train_loader_iter) 
                   
        inputs1, inputs2 = inputs1.to(args.local_rank, non_blocking=True), inputs2.to(args.local_rank, non_blocking=True)
        cls_labels = cls_labels.to(args.local_rank, non_blocking=True)
        inputs = torch.cat([inputs1, inputs2], dim=0)
        img_box = torch.cat([img_box1, img_box2], dim=0)
        cls_labels_double = torch.cat([cls_labels, cls_labels], dim=0)
        
        cls, segs, feature_map, z = ugrl(inputs, seg_detach=args.seg_detach)
        
        cls1, cls2 = torch.chunk(cls, 2, dim=0)
        segs1, segs2 = torch.chunk(segs, 2, dim=0)
        feature_map1, feature_map2 = torch.chunk(feature_map, 2, dim=0)
        z1, z2 = torch.chunk(z, 2, dim=0)
        
        cams = multi_scale_cam(ugrl.module, inputs=inputs, scales=cfg.cam.scales)
        cams1, cams2 = torch.chunk(cams, 2, dim=0)
        
        valid_cam1, pseudo_label1 = cam_to_label(cams1.detach(), cls_label=cls_labels, img_box=img_box1, ignore_mid=True, cfg=cfg)
        valid_cam2, pseudo_label2 = cam_to_label(cams2.detach(), cls_label=cls_labels, img_box=img_box2, ignore_mid=True, cfg=cfg)
        
        segs1 = F.interpolate(segs1, size=pseudo_label1.shape[1:], mode='bilinear', align_corners=False)
        segs2 = F.interpolate(segs2, size=pseudo_label2.shape[1:], mode='bilinear', align_corners=False)
        
        segs1 = F.interpolate(segs1, size=pseudo_label1.shape[1:], mode='bilinear', align_corners=False)
        segs2 = F.interpolate(segs2, size=pseudo_label2.shape[1:], mode='bilinear', align_corners=False)
        
        seg_loss1 = get_seg_loss(segs1, pseudo_label1.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_loss2 = get_seg_loss(segs2, pseudo_label2.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_loss = (seg_loss1 + seg_loss2) / 2.0
        
        cls_loss_per_sample = F.multilabel_soft_margin_loss(cls, cls_labels_double, reduction='none')
        cls_loss_per_sample1, cls_loss_per_sample2 = torch.chunk(cls_loss_per_sample, 2, dim=0)
        
        with torch.no_grad():
            uncertainty_scores1 = uncertainty_calculator(feature_map1, cls_labels)
            uncertainty_scores2 = uncertainty_calculator(feature_map2, cls_labels)
        
        modulated_cls_loss1 = torch.mean((1 - uncertainty_scores1.unsqueeze(1)) * cls_loss_per_sample1)
        modulated_cls_loss2 = torch.mean((1 - uncertainty_scores2.unsqueeze(1)) * cls_loss_per_sample2)
        modulated_cls_loss = (modulated_cls_loss1 + modulated_cls_loss2) / 2.0

        cls_loss = torch.mean(cls_loss_per_sample)
        
        vicreg_total_loss, vicreg_inv_loss, vicreg_var_loss, vicreg_cov_loss = vicreg_criterion(z1, z2)
        
        if n_iter <= cfg.train.cam_iters:
            loss = 1.0 * modulated_cls_loss + cfg.vicreg.initial_weight * vicreg_total_loss
        else: 
            loss = 1.0 * modulated_cls_loss + 0.1 * seg_loss + cfg.vicreg.weight * vicreg_total_loss

        avg_meter.add({'cls_loss': cls_loss.item(), 
                       'modulated_cls_loss': modulated_cls_loss.item(),
                       'seg_loss': seg_loss.item(),
                       'vicreg_total_loss': vicreg_total_loss.item(),
                       'vicreg_inv_loss': vicreg_inv_loss.item(),
                       'vicreg_var_loss': vicreg_var_loss.item(),
                       'vicreg_cov_loss': vicreg_cov_loss.item()})
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (n_iter+1) % cfg.train.log_iters == 0:
            if dist.get_rank() == 0:
                delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
                cur_lr = optimizer.param_groups[0]['lr']
                
                preds = torch.argmax(segs1,dim=1,).cpu().numpy().astype(np.int16)
                gts = pseudo_label1.cpu().numpy().astype(np.int16)
                
                seg_mAcc = (preds==gts).sum()/preds.size
                
                grid_imgs, grid_cam = imutils.tensorboard_image(imgs=inputs1.clone(), cam=valid_cam1)

                grid_labels = imutils.tensorboard_label(labels=gts)
                grid_preds = imutils.tensorboard_label(labels=preds)
                
                cls_loss_val = avg_meter.pop('cls_loss')
                modulated_cls_loss_val = avg_meter.pop('modulated_cls_loss')
                seg_loss_val = avg_meter.pop('seg_loss')
                
                vicreg_total_loss_val = avg_meter.pop('vicreg_total_loss')
                vicreg_inv_loss_val = avg_meter.pop('vicreg_inv_loss')
                vicreg_var_loss_val = avg_meter.pop('vicreg_var_loss')
                vicreg_cov_loss_val = avg_meter.pop('vicreg_cov_loss')
                
                log_message = (
                    f"Iter: {n_iter+1}; Elasped: {delta}; ETA: {eta}; LR: {cur_lr:.3e}; "
                    f"cls_loss: {cls_loss_val:.4f}; "
                    f"modulated_cls_loss: {modulated_cls_loss_val:.4f}; "
                    f"pseudo_seg_loss: {seg_loss_val:.4f}; "
                    f"pseudo_seg_mAcc: {seg_mAcc:.4f} "
                    "| "
                    f"vicreg_total_loss: {vicreg_total_loss_val:.4f} "
                    f"vicreg_inv_loss: {vicreg_inv_loss_val:.4f} "
                    f"vicreg_var_loss: {vicreg_var_loss_val:.4f} "
                    f"vicreg_cov_loss: {vicreg_cov_loss_val:.4f}"
                )
                logging.info(log_message)

                writer.add_image("train/images", grid_imgs, global_step=n_iter)
                writer.add_image("train/preds", grid_preds, global_step=n_iter)
                writer.add_image("train/pseudo_gts", grid_labels, global_step=n_iter)
                
                writer.add_image("cam/valid_cams", grid_cam, global_step=n_iter)

                writer.add_scalars('train/loss', {
                    "seg_loss": seg_loss_val, 
                    "cls_loss": cls_loss_val,
                    "modulated_cls_loss": modulated_cls_loss_val,
                    "vicreg_total_loss": vicreg_total_loss_val,
                    "vicreg_inv_loss": vicreg_inv_loss_val,
                    "vicreg_var_loss": vicreg_var_loss_val,
                    "vicreg_cov_loss": vicreg_cov_loss_val,
                }, global_step=n_iter)
                
                writer.add_scalar('train/mean_uncertainty', uncertainty_scores1.mean().item(), global_step=n_iter)
            
        if (n_iter+1) % cfg.train.eval_iters == 0:
            if dist.get_rank() == 0:
                ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "ugrl_iter_%d.pth"%(n_iter+1))
                
                logging.info('Saving checkpoint...')
                torch.save(ugrl.module.state_dict(), ckpt_name)
                
                logging.info('Validating...')
                cls_score, seg_score, cam_score = validate(model=ugrl, data_loader=val_loader, cfg=cfg)
                
                logging.info("val cls score: %.6f"%(cls_score))
                logging.info("cams score:\n%s"%cam_score)
                logging.info("segs score:\n%s"%seg_score)
    return True

if __name__ == "__main__":
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size
    
    cfg.cam.high_thre = args.high_thre
    cfg.cam.low_thre = args.low_thre

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    if args.local_rank == 0:
        os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
        os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
        os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)
        
        setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
        logging.info('\nargs: %s' % args)
        logging.info('\nconfigs: %s' % cfg)
    
    setup_seed(1)
    train(cfg=cfg)