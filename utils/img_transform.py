from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
# import torch.nn as nn
# from torch.autograd import Variable
# import _init_paths
# import nn as mynn
# from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
# from core.test import im_detect_all, im_detect_regress
# from modeling.model_builder import Generalized_RCNN
# import datasets.dummy_datasets as datasets
# import utils.misc as misc_utils
# import utils.net as net_utils
# import utils.vis as vis_utils
# from utils.detectron_weight_helper import load_detectron_weight
# from utils.timer import Timer

import cv2 

import torch  
import torchvision.transforms as T
torch.manual_seed(2)

# def configure_detector(data, pred_score, gpu_id, set_cfgs = None, cuda=True, load_detectron=False):
#     if not torch.cuda.is_available():
#         sys.exit("Need a CUDA device to run the code.")
#     torch.cuda.set_device(gpu_id)
#     torch.set_num_threads(1)
#     if data == 'coco':
#         dataset = datasets.get_coco_dataset()
#         cfg.MODEL.NUM_CLASSES = len(dataset.classes)

#     elif data == 'clasp2020':
#         dataset = datasets.get_clasp_dataset()
#         cfg.MODEL.NUM_CLASSES = len(dataset.classes)

#     elif data in ['clasp1_2021', 'clasp2_2021']:
#         dataset = datasets.get_clasp1_dataset()
#         cfg.MODEL.NUM_CLASSES = len(dataset.classes)

#     elif data == 'mot_2020':
#         dataset = datasets.get_clasp_dataset()
#         cfg.MODEL.NUM_CLASSES = len(dataset.classes)

#     elif data == 'AppleA_train':
#         dataset = datasets.get_AppleA_dataset()
#         cfg.MODEL.NUM_CLASSES = len(dataset.classes)

#     elif data=="keypoints_coco":
#         dataset = datasets.get_coco_dataset()
#         cfg.MODEL.NUM_CLASSES = 2
#     else:
#         raise ValueError('Unexpected dataset name: {}'.format(data))

#     print('load cfg from file: {}'.format(cfg_file))
#     cfg_from_file(cfg_file)
#     # set NMS
#     cfg['TEST']['NMS'] = 0.3
#     cfg['TEST']['SCORE_THRESH'] = pred_score
#     if set_cfgs is not None:
#         cfg_from_list(set_cfgs)

#     assert bool(load_ckpt) ^ bool(load_detectron), \
#         'Exactly one of --load_ckpt and --load_detectron should be specified.'
#     cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
#     assert_and_infer_cfg()

#     maskRCNN = Generalized_RCNN()

#     if cuda:
#         maskRCNN.cuda()

#     if load_ckpt:
#         load_name = load_ckpt
#         print("loading checkpoint %s" % (load_name))
#         checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
#         net_utils.load_ckpt(maskRCNN, checkpoint['model'], ckpt_frcnn=False, isTrain=False)

#     if load_detectron:
#         print("loading detectron weights %s" % load_detectron)
#         load_detectron_weight(maskRCNN, load_detectron, isTrain=False)

#     maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
#                                  minibatch=True, device_ids=[gpu_id])  # only support single GPU

#     maskRCNN.eval()

#     return maskRCNN, dataset

storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62'
labeled_img_dir = f'{storage}/SoftTeacher/data/flower/unlabeledFlower/00150000.png'
labeled_img_dir = f'{storage}/tracking_wo_bnw/data/CLASP1/train_gt_det/img1/01400000.png'
out_dir = f'{storage}/SoftTeacher/data/clasp1/color_augmented_img'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

plt.rcParams["savefig.bbox"] = 'tight'
orig_img = Image.open(labeled_img_dir)

def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'
    load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'  # baseline
    data = 'coco'
    pred_score = 0.5
    gpu_id = 0
    class_list = list(np.arange(1,81))#[1]#, 25, 27, 29] #list(np.arange(1,81)) #[1, 25, 27, 29]
    load_detectron = None
    # maskRCNN, dataset = configure_detector(data, pred_score, gpu_id)
        
    jitter = T.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.2, hue=0.05)
    jitted_imgs = [jitter(orig_img) for _ in range(3)]
    fr = int(os.path.basename(labeled_img_dir).split('.')[0])
    
    #plot(jitted_imgs)

    # autocontraster = T.RandomAutocontrast()
    # autocontrasted_imgs = [autocontraster(orig_img) for _ in range(4)]

    # sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=2)
    # sharpened_imgs = [sharpness_adjuster(orig_img) for _ in range(4)]

    # rotater = T.RandomRotation(degrees=(0, 180))
    # rotated_imgs = [rotater(orig_img) for _ in range(4)]
    angle = 0
    im_ind = 0
    for i, imc in enumerate(jitted_imgs):

        blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        blurred_imgs = [blurrer(imc) for _ in range()]
        blurred_imgs.insert(0, imc)
        for im in blurred_imgs:
            imgname = f'{out_dir}/{fr}_{im_ind}.png'
            im_ind+=1
            
            im.save(imgname ,"PNG")
            # im = cv2.imread(imgname)
            # imgname = f'{out_dir}/{fr}_{i}.png'
            # cls_boxes, \
            # cls_segms, \
            # cls_keyps, \
            # cls_segms_coarse = im_detect_all(maskRCNN, im,
            #                                     test_aug=0, soft_nms=1,
            #                                     nms_thr=0.5,
            #                                     score_thr=0.5)
            # vis_utils.vis_clasp(
            #                     fr,
            #                     angle,
            #                     im[:, :, ::-1],  # BGR -> RGB for visualization
            #                     imgname,
            #                     out_dir,
            #                     cls_boxes,
            #                     cls_segms,
            #                     cls_segms_coarse,
            #                     cls_keyps,
            #                     dataset=dataset,
            #                     class_list=class_list,
            #                     box_alpha=1,
            #                     show_class=True,
            #                     thresh=pred_score,
            #                     kp_thresh=2,
            #                     show_mask=1
            #                 )