#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:05:16 2021

@author: Siddique
"""

import os
import sys
from torch.utils.tensorboard import SummaryWriter
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch
import pdb
#%%
# if your dataset is in COCO format, register dataset in coco format:
#from detectron2.data.datasets import register_coco_instances
#from detectron2.data.datasets import register_coco_panoptic
from detectron2.data.datasets import register_coco_panoptic_separated
import argparse
coderoot = os.path.dirname(os.path.realpath(__file__)).split('ssl_flower_semantic')[0] + 'ssl_flower_semantic'
print(f'coderoot:{coderoot}')
sys.path.insert(0, f"{coderoot}")

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--database', dest='database', required=True,
        help='Database to use: clasp1 or clasp2', default='clasp2')

    parser.add_argument(
    '--data_set', dest='data_set', required=True,
    help='unlabeled dataset to use: AppleA or AppleB or Peach or Pear', default='AppleA')

    parser.add_argument(
        '--label_percent',
        help='percent of manual annotations used in semi-SL',
        default=1, type=int)
    
    parser.add_argument(
        '--number_gpus',
        help='percent of manual annotations used in semi-SL',
        default=1, type=int)
    
    parser.add_argument(
        '--CV',
        help='Cross Validation in SSL iteration',
        default=1, type=int)
    
    parser.add_argument(
        '--ssl_iter',
        help='SSL iteration index to make sure the pretrained model is loaded properly',
        default=0, type=int)
    parser.add_argument(
        '--lambda_sem',
        help='(1-lambda_sem)*(lc+lb+lm) + lambda_sem*ls',
        default=0.5, type=float)
    
    parser.add_argument(
        '--gpu_id',
        help='single gpu training',
        default=0, type=int)

    parser.add_argument(
        '--pretrained',
        help='single gpu training',
        default=0, type=int)

    parser.add_argument('--model_type', type=str, default='SSL')

    parser.add_argument('--working_dir', type=str, default='/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--nw', dest='num_workers',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        type=int)
    parser.add_argument(
        '--iter_size',
        help='Update once every iter_size steps, as in Caffe.',
        default=1, type=int)

    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)
    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)

    return parser.parse_args()

def load_labeled_data(args, cfg):
    storage = f'{args.working_dir}/tracking_wo_bnw/data/flower/train_gt_panoptic_sw'

    img_dir = f'{storage}/trainFlowerAug'
    panoptic_masks = f'{storage}/panoptic_labels'
    panoptic_json = f'{storage}/panoptic_train_2021.json'

    #semantic mask> 0: flower, 255: background
    sem_seg_masks = f'{storage}/semantic_labels'
    instance_json = f'{storage}/instances_train_2021.json'

    register_coco_panoptic_separated(name="flower_panoptic_labeled",
                                    metadata={},
                                    image_root=img_dir,
                                    panoptic_root=panoptic_masks,
                                    panoptic_json=panoptic_json,
                                    sem_seg_root=sem_seg_masks,
                                    instances_json=instance_json,
                                    gt_ext="png",image_ext="png" )
    cfg.DATASETS.TRAIN = ("AppleA_train",)
    return cfg
#torch.cuda.set_device(1)
#torch.set_num_threads(1)
#meta={'thing_dataset_id_to_contiguous_id': {1:1}}
args = parse_args()
print('Called with args:')
print(args)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.DEVICE = args.gpu_id

if args.database=='coco':
    coco_path = '/media/NAS/LabFiles/CLASP/CLASP_Data/Data_GT/coco_images/coco_2017'
    img_dir = coco_path + '/val2017'
    panoptic_json = coco_path + '/panoptic_annotations_trainval2017/annotations/panoptic_val2017.json'
    panoptic_masks = coco_path + '/panoptic_annotations_trainval2017/annotations/panoptic_val2017/panoptic_val2017'
    stuff_masks = coco_path + '/stuff_annotations_trainval2017/annotations/stuff_val2017_pixelmaps'
    instance_json = coco_path + '/annotations/instances_val2017.json'

    register_coco_panoptic_separated(name="val_panoptic",
                                    metadata={},
                                    image_root=img_dir,
                                    panoptic_root=panoptic_masks,
                                    panoptic_json=panoptic_json,
                                    sem_seg_root=stuff_masks,
                                    instances_json=instance_json)
    cfg.DATASETS.TRAIN = ("val_panoptic_separated",)

if args.database=='flower' and args.model_type == 'SL':
    cfg = load_labeled_data(args, cfg)
    storage = f'{coderoot}/dataset/ssl_train/train_gt_panoptic_sw_16_8'

    img_dir = f'{storage}/trainFlowerAug'
    panoptic_masks = f'{storage}/panoptic_labels'
    panoptic_json = f'{storage}/panoptic_train_2021.json'

    #semantic mask> 0: flower, 255: background
    sem_seg_masks = f'{storage}/semantic_labels'
    instance_json = f'{storage}/instances_train_2021.json'


elif args.database=='flower' and args.model_type == 'SSL':
    cfg = load_labeled_data(args, cfg)
    storage = f'{coderoot}/dataset/ssl_train/aug_gt_pan/iter{args.ssl_iter}'
    
    if args.number_gpus>2: # use local storage in remote server
        storage = f"/media/siddique/6TB2/aug_gt_pan/iter{args.ssl_iter}"
    print(f'>>>> load training data: {storage}')
       
    img_dir = f'{storage}/img1_{args.ssl_iter}'
    panoptic_masks = f'{storage}/panoptic_labels'
    panoptic_json = f'{storage}/panoptic_flower_test_aug_{args.ssl_iter}.json'

    #semantic mask> 0: flower, 255: background
    sem_seg_masks = f'{storage}/semantic_labels'
    instance_json = f'{storage}/instances_flower_test_aug_{args.ssl_iter}.json'
else:
    print('model type should be SL or SSL')

register_coco_panoptic_separated(name="flower_panoptic_unlabeled",
                                metadata={},
                                image_root=img_dir,
                                panoptic_root=panoptic_masks,
                                panoptic_json=panoptic_json,
                                sem_seg_root=sem_seg_masks,
                                instances_json=instance_json,
                                gt_ext="png",image_ext="png" )
cfg.DATASETS.TRAIN = ("flower_panoptic_unlabeled_separated",)#,"flower_panoptic_labeled_separated" #


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


cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#for iter0 only
if args.ssl_iter==0:
    cfg.MODEL.WEIGHTS = f'{coderoot}/models/{args.model_type}/AppleA_train/model_0019999.pth'
    model_dir = f'{coderoot}/models/{args.model_type}/AppleA_train'
else:
    #TODO: loss is not yet modified
    #previous model
    model_dir = f'{coderoot}/models/{args.model_type}/{args.data_set}'
    if args.ssl_iter==1:
        cfg.MODEL.WEIGHTS = f'{coderoot}/models/SL/AppleA_train/model_0019999.pth'
    else:
        cfg.MODEL.WEIGHTS = f'/{model_dir}/CV{args.CV}/iter{args.ssl_iter-1}/model_0024999.pth'
        
    if not os.path.exists(cfg.MODEL.WEIGHTS):
        cfg.MODEL.WEIGHTS = f'/{model_dir}/CV{args.CV}/iter{args.ssl_iter-1}/model_0019999.pth'
    if not os.path.exists(cfg.MODEL.WEIGHTS):
        cfg.MODEL.WEIGHTS = f'/{model_dir}/CV{args.CV}/iter{args.ssl_iter-1}/model_0014999.pth'

if args.pretrained:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

print(f'loaded weights: {cfg.MODEL.WEIGHTS}')
# Directory where output files are written
cfg.OUTPUT_DIR = f'{model_dir}/CV{args.CV}/iter{args.ssl_iter}'
if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025 #1 gpu: 0.0025, 2 gpu: 0.005
#weight decay: 0.0001
cfg.SOLVER.MAX_ITER = 20000
cfg.SOLVER.STEPS = (2000, 5000, 10000)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
if args.database=='coco':
    print('cfg.MODEL.ROI_HEADS.NUM_CLASSES: {}'.format(cfg.MODEL.ROI_HEADS.NUM_CLASSES))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 92
    print('cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES: {}'.format(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES))
else:
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
    #MTL weight
    #TODO: instead of tuning manually, update the weight based on the task variance
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = args.lambda_sem #default:0.5
    cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT = (1 - args.lambda_sem) #default: 0.1
    print(f'sem loss weight: {cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT}, instance loss weight: {cfg.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT}')
cfg.MODEL.BACKBONE.FREEZE_AT = 5
cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800, 1080, 1152)
#TODO: control the anchor generation to handle the small flower detection
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]] 
cfg.VIS_PERIOD = 1000
#pdb.set_trace()
#TODO: is it possible to modify the loss function using default trainer??

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
