import torch
import torch.nn.functional as F
from .imutils import denormalize_img
from .dcrf import crf_inference_label
import numpy as np

def cam_to_label(cam, cls_label, img_box=None, ignore_mid=False, cfg=None):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value<=cfg.cam.bkg_score] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value<=cfg.cam.high_thre] = cfg.dataset.ignore_index
        _pseudo_label[cam_value<=cfg.cam.low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * cfg.dataset.ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return valid_cam, pseudo_label

def ignore_img_box(label, img_box, ignore_index):

    pseudo_label = torch.ones_like(label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return pseudo_label

def cam_to_fg_bg_label(imgs, cams, cls_label, bg_thre=0.3, fg_thre=0.6):

    scale = 2
    imgs = F.interpolate(imgs, size=(imgs.shape[2]//scale, imgs.shape[3]//scale), mode="bilinear", align_corners=False)
    cams = F.interpolate(cams, size=imgs.shape[2:], mode="bilinear", align_corners=False)

    b, c, h, w = cams.shape
    _imgs = denormalize_img(imgs=imgs)

    cam_label = torch.ones(size=(b, h, w),).to(cams.device)
    bg_label = torch.ones(size=(b, 1),).to(cams.device)
    _cls_label = torch.cat((bg_label, cls_label), dim=1)

    lt_pad = torch.ones(size=(1, h, w),).to(cams.device) * bg_thre
    ht_pad = torch.ones(size=(1, h, w),).to(cams.device) * fg_thre

    for i in range(b):
        keys = torch.nonzero(_cls_label[i,...])[:,0]
        #print(keys)
        n_keys = _cls_label[i,...].cpu().numpy().sum().astype(np.uint8)
        valid_cams = cams[i, keys[1:]-1, ...]
        
        lt_cam = torch.cat((lt_pad, valid_cams), dim=0)
        ht_cam = torch.cat((ht_pad, valid_cams), dim=0)

        _, cam_label_lt = lt_cam.max(dim=0)
        _, cam_label_ht = ht_cam.max(dim=0)
        #print(_imgs[i,...].shape)
        _images = _imgs[i,...].permute(1,2,0).cpu().numpy().astype(np.uint8)
        _cam_label_lt = cam_label_lt.cpu().numpy()
        _cam_label_ht = cam_label_ht.cpu().numpy()
        _cam_label_lt_crf = crf_inference_label(_images, _cam_label_lt, n_labels=n_keys)
        _cam_label_lt_crf_ = keys[_cam_label_lt_crf]
        _cam_label_ht_crf = crf_inference_label(_images, _cam_label_ht, n_labels=n_keys)
        _cam_label_ht_crf_ = keys[_cam_label_ht_crf]
        #_cam_label_lt_crf = torch.from_numpy(_cam_label_lt_crf).to(cam_label.device)
        #_cam_label_ht_crf = torch.from_numpy(_cam_label_ht_crf).to(cam_label.device)
        
        cam_label[i,...] = _cam_label_ht_crf_
        cam_label[i, _cam_label_ht_crf_==0] = 255
        cam_label[i, (_cam_label_ht_crf_ + _cam_label_lt_crf_)==0] = 0
        #imageio.imsave("out.png", encode_cmap(cam_label[i,...].cpu().numpy()))
        #cam_label_lt

    return cam_label

def multi_scale_cam(model, inputs, scales):
    cam_list = []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam, _ = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        
        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam, _ = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam
