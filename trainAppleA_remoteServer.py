#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:05:16 2021

@author: Siddique
"""

import os
from torch.utils.tensorboard import SummaryWriter
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch
#%%
# if your dataset is in COCO format, register dataset in coco format:
#from detectron2.data.datasets import register_coco_instances
#from detectron2.data.datasets import register_coco_panoptic
from detectron2.data.datasets import register_coco_panoptic_separated

#torch.cuda.set_device(1)
#torch.set_num_threads(1)
#meta={'thing_dataset_id_to_contiguous_id': {1:1}}
dataset = 'AppleA_train' #'AppleA'
if dataset=='coco':
    coco_path = '/media/NAS/LabFiles/CLASP/CLASP_Data/Data_GT/coco_images/coco_2017'
    img_dir = coco_path + '/val2017'
    panoptic_json = coco_path + '/panoptic_annotations_trainval2017/annotations/panoptic_val2017.json'
    panoptic_masks = coco_path + '/panoptic_annotations_trainval2017/annotations/panoptic_val2017/panoptic_val2017'
    stuff_masks = coco_path + '/stuff_annotations_trainval2017/annotations/stuff_val2017_pixelmaps'
    instance_json = coco_path + '/annotations/instances_val2017.json'

if dataset == 'AppleA_train':
    NAS = '/media/NAS/LabFiles/Walden'
    base_dir = NAS + '/trainTestSplit/train/dataFormattedProperly/splitImages4x3'

    img_dir = base_dir + '/flowersSplit'
    panoptic_masks = base_dir + '/rgbMasksSplit'
    panoptic_json = base_dir + '/appleA_panoptic_split4x3_train.json'

    # 0: flower, 255: background
    sem_seg_masks = base_dir + '/semMasksSplit'
    instance_json = base_dir + '/appleA_instance_split4x3_train.json'


# =============================================================================
# from detectron2.data import DatasetCatalog
# DatasetCatalog.get('AppleA_pan')
# =============================================================================
#%%
#from detectron2.data import MetadataCatalog
#MetadataCatalog.get("AppleA_pan").thing_classes = ['flower']
#MetadataCatalog.get("AppleA_pan").stuff_classes = ['background']
#MetadataCatalog.get("AppleA_pan").thing_dataset_id_to_contiguous_id = {1:1}
#MetadataCatalog.get("AppleA_pan").stuff_dataset_id_to_contiguous_id = {0:0}
#MetadataCatalog.get("AppleA_pan").panoptic_json = panoptic_json


#%%
'''
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
'''
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.DEVICE = 1
if dataset=='coco':
    register_coco_panoptic_separated(name="val_panoptic",
                                     metadata={},
                                     image_root=img_dir,
                                     panoptic_root=panoptic_masks,
                                     panoptic_json=panoptic_json,
                                     sem_seg_root=stuff_masks,
                                     instances_json=instance_json)
    cfg.DATASETS.TRAIN = ("val_panoptic_separated",)
else:
    register_coco_panoptic_separated(name="AppleA_split",
                                     metadata={},
                                     image_root=img_dir,
                                     panoptic_root=panoptic_masks,
                                     panoptic_json=panoptic_json,
                                     sem_seg_root=sem_seg_masks,
                                     instances_json=instance_json)
    cfg.DATASETS.TRAIN = ("AppleA_split_separated",)

cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#for iter0 only
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)

# Directory where output files are written
cfg.OUTPUT_DIR = "/home/abubakarsiddique/trainPanopticModel/flower_model"
if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 30000
cfg.SOLVER.STEPS = (2000, 5000, 10000)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
if dataset=='coco':
    print('cfg.MODEL.ROI_HEADS.NUM_CLASSES: {}'.format(cfg.MODEL.ROI_HEADS.NUM_CLASSES))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 92
    print('cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES: {}'.format(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES))
else:
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
    #MTL weight
    #TODO: instead of tuning manually, update the weight based on the task variance
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0 #default:0.5
    cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT = 0.5 #default: 0.1

cfg.MODEL.BACKBONE.FREEZE_AT = 4
cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800, 1080, 1152)
#TODO: control the anchor generation to handle the small flower detection
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]] 
cfg.VIS_PERIOD = 1000
#TODO: is it possible to modify the loss function using default trainer??
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
