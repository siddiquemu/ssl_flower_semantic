#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed January 7 2022

@author: Abubakar Siddique
"""

from __future__ import division
from __future__ import print_function
import copy
from functools import total_ordering
import glob
import os
import sys
import warnings
coderoot = os.path.dirname(os.path.realpath(__file__)).split('ssl_flower_semantic')[0] + 'ssl_flower_semantic'
print(f'coderoot:{coderoot}')
sys.path.insert(0, f"{coderoot}")
sys.path.insert(0, f'{coderoot}/rgr-public/Python')
from runRGR import RGR

import argparse

import numpy as np
import itertools
import cv2
import time
import torch
import imutils
import pycocotools.mask as mask_util
from skimage import measure
import imageio
import glob
import torch.nn as nn
from torch.autograd import Variable   
# from get_cluster_mode import Cluster_Mode
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import multiprocessing as mp
from itertools import islice
from utils.init_pseudo_labels_dir import get_all_dirs, delete_all
from pycocotools.coco import COCO
from utils.clasp2coco import init_annotations_info_pan, Write_To_Json, Write_Segments_info_pan, \
    Write_ImagesInfo, Write_AnnotationInfo, load_clasp_json, get_frame_anns, define_dataset_dictionary_pan, \
    Proxy2RealObj, define_dataset_dictionary_mp
from skimage.morphology import label
from src.panoptic_flower_model import panoptic_fpn_flower
from PIL import Image
from utils.box_jitter import aug_box_jitter

class Pseudo_Labels(panoptic_fpn_flower):
    # goal is to reduce pseudo-labels computation time by using the number of subprocess
    def __init__(self, gpu_id, init_params,  img_subset, ann_ids, frame_ids, all_rgb, dataset_flower_ins, dataset_flower_pan,
                p_id=None, verbose=True):
        init_params['gpu_id'] = gpu_id
        super(Pseudo_Labels, self).__init__(init_params)
        self.init_detector()

        self.gpu_id = gpu_id
        self.p_id = p_id
        self.data = init_params['data']
        self.init_params = init_params
        self.remap_score_thr = init_params['remap_score_thr']
        self.load_ckpt_frcnn = None
        self.maskRCNN = None
        self.img_subset = img_subset
        self.verbose = verbose
        self.database = init_params['database']
        self.vis_annos = None
        self.vis_dir = None
        self.vis_dir = init_params['storage_dir']
        self.result_path = init_params['output_dir']
        self.det_thr = self.init_params['pred_score']
        self.nms_thr = self.init_params['nms_thr']
        self.regress_pred_score = init_params['regress_pred_score']
        self.class_list = self.init_params['class_list']
        self.angleSet = self.init_params['angleSet']
        self.regress_cluster = self.init_params['regress_aug_prop']
        self.save_dets_dict = {}
        self.save_dets_dict['dets_aug'] = open(
            os.path.join(self.result_path, '{}'.format('all_cam' + '_pb_1aug1nms.txt')), mode='w')

        self.detPB = []
        self.vis_path = os.path.join(self.result_path, 'vis')
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        else:
            delete_all(self.vis_path)

        self.cluster_score_thr = self.init_params['cluster_score_thr']
        self.semi_supervised = self.init_params['semi_supervised']
        self.apply_cluster_mode = self.init_params['apply_cluster_mode']
        self.ranges = init_params['angle_ranges']


        #global variable
        #self.ann_id_count = ann_ids
        self.ann_ids = ann_ids
        #self.ann_id_count.append(1)
        self.semi_frames = init_params['semi_frames']
        self.dataset_flower_ins = dataset_flower_ins
        self.dataset_flower_pan = dataset_flower_pan
        self.frame_ids = frame_ids
        self.labeled_frames = init_params['labeled_frames']

        self.instance_label_path = init_params['sem_seg_masks']
        self.panoptic_label_path = init_params['panoptic_masks']
        self.colorset = init_params['colorset']
        self.all_rgb = all_rgb
        self.num_sem_class = init_params['num_sem_class']
        self.softmax = nn.Softmax(dim=0)
        
        if self.init_params['pretrained']:
            self.flower_class_id = 9
        else:
            self.flower_class_id = 0

    @staticmethod
    def get_random_angles(ranges=None, factor=None):
        angleSet = [0, 180]
        for intvl in ranges:
            angleSet += random.sample(range(intvl[0], intvl[1]), factor)
        return angleSet

    @staticmethod
    def rotated_mask(mask, angle=None):
        #mask[mask>0] = 255
        mask_rot = imutils.rotate_bound(mask, angle)
        mask_rot[mask_rot>0] = 255
        return mask_rot
    
    @staticmethod
    def rgr_refine(pred, img):
        """
        Args:
            pred: raw normalized scoremap from test-time augmentation procedure (num_class+1, H, W)
            img: RGB input window (H, W, C) >>> C=3
        """
        #call RGR
        print(f'refining using RGR...')
        pred = np.transpose(pred, (1,2,0))
        soft_mask = np.zeros(pred.shape, dtype='float')
        
        #Peach
        #pred[:,:,0][pred[:,:,0]<0.4] = 0.00001
        # soft_mask[:,:,0] = 1-pred[:,:,0]
        # soft_mask[:,:,1] = pred[:,:,0]
        #Pear
        #pred[:,:,0][pred[:,:,0]<0.5] = 0.001
        soft_mask[:,:,0] = pred[:,:,1]
        soft_mask[:,:,1] = pred[:,:,0]
        assert img.shape[:2]==pred.shape[:2]
        
        # fixed parameters
        numSets = 10    # number of seeds sets (samplings)
        cellSize = 100   # average spacing between samples

        ## RGR parameters
        # thresholds
        tau0 = 0.5  # original CNN threshold
        tauF = 0.7  # high confidence foreground
        tauB = 0.01     # high confidence background
        m = 0.1
        warnings.filterwarnings("ignore")
        im_color, finalMask = RGR(img, soft_mask, m, numSets, cellSize, tau0, tauF, tauB)
        finalMask = cv2.cvtColor(finalMask, cv2.COLOR_RGB2GRAY)
        finalMask[finalMask>0]=1
        return finalMask

    @staticmethod
    def init_fig(im):
        im = im[:, :, ::-1]
        warnings.filterwarnings("ignore")
        fig = plt.figure(frameon=False)
        fig.set_size_inches(im.shape[1] / 200, im.shape[0] / 200)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(im)
        return ax, fig


    @staticmethod
    def vis_gt(im, boxs, masks=None, gt_vis_path=None, imname=None):
        #implement pseudo-labels mask visualization as well
        for i, bb in enumerate(boxs):
            im = cv2.rectangle(im, (np.int(bb[2]), np.int(bb[3])), (np.int(bb[2]+bb[4]), np.int(bb[3]+bb[5])), (0,255,0), 4)
            if masks is not None:
                mask = masks[i]
                if isinstance(mask, dict):
                    mask = mask_util.decode(mask)
                else:
                    mask = np.array(mask, dtype='uint8')
                #contours = measure.find_contours(bmask, 0.5)
                contours,_= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    im = cv2.drawContours(im, contour, -1, (255, 0, 0), 4)
        # cv2.putText(imgcv, classGT, (int(xmin+w/2), int(ymin+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        cv2.imwrite(os.path.join(gt_vis_path, os.path.basename(imname)), im)


    @staticmethod
    def get_gt_labels(fr, anns_fr, coco_json=None, angle=None):
        """segms are Polygon format in anns_fr
        fr_mask will be returned in rle format
        """
        fr_det = []
        fr_mask = []
        assert coco_json is not None
        #print(f'total flower found in GT {len(anns_fr)}')
        #print(f'labeled frame: {fr} angle: {angle}')
        for i, ann in enumerate(anns_fr):
            if angle>0:
                try:
                    label_mask = coco_json.annToMask(ann)
                except:
                    print(ann)
                    continue
                label_mask_rot = imutils.rotate_bound(label_mask, angle)
                label_mask_rot[label_mask_rot>0] = 255
                rle = mask_util.encode(np.asfortranarray(label_mask_rot))
                fr_mask.append(rle)
                #print(f'{i}: Bounding box = {mask_util.toBbox(rle)}')
                x,y,w,h = mask_util.toBbox(rle)
                fr_det.append([fr, i, x, y, w, h, 1.,1, angle])
            else:
                #print(ann)
                try:
                    rle = coco_json.annToRLE(ann)
                except:
                    print(ann)
                    continue
                fr_mask.append(rle)
                x,y,w,h = mask_util.toBbox(rle)
                fr_det.append([fr, i, x, y, w, h, 1.,1, angle])
        #assert len(fr_det)==len(fr_mask)>0
        return fr_det, fr_mask
    
    def boxes_regression(self, im, fr_boxes, jitter_boxes=False, 
    jitter_factor=0.1, jitter_size=20, score_thr=0.4, show_jitter=False):
        if fr_boxes.ndim==1:
            print(fr_boxes)
            fr_boxes = np.reshape(fr_boxes, (1, fr_boxes.shape[0]))

        aug_proposals = torch.tensor(fr_boxes[:, 2:6], device=self.gpu_id)
        
        if jitter_boxes:
            aug_proposals = torch.reshape(aug_proposals, (len(fr_boxes),1,4))
            aug_proposals = aug_box_jitter(aug_proposals, times=jitter_size, frac=jitter_factor) # size: (len(fr_boxes),10,1,4)
            aug_proposals = torch.stack(aug_proposals, dim=0)
            aug_proposals = torch.reshape(aug_proposals, (len(fr_boxes)*jitter_size, 4))

        if not show_jitter:
            pred_results = self.predict_single(im, pre_proposals=aug_proposals, apply_jitter=True)
            scores = pred_results.scores.cpu().numpy()
            print(scores)
            keep = scores>=score_thr

            boxs = pred_results.pred_boxes.tensor.cpu().numpy()
            boxs = boxs[keep]
            boxs[:,2:4] -= boxs[:,0:2]
            boxs = boxs.astype('int')

            masks = pred_results.pred_masks.cpu().numpy()
            masks = masks[keep]
            scores = scores[keep]
            
            regressed_boxes = np.ones((len(boxs), fr_boxes.shape[-1]))
            regressed_boxes[:,2:6] = boxs
        else:
            regressed_boxes = np.ones((len(aug_proposals), fr_boxes.shape[-1]))
            regressed_boxes[:, 2:6] = aug_proposals.cpu().numpy()
            masks = None
        return regressed_boxes, masks

    def gen_unique_color(self):
        unique=False
        while not unique:
                color = tuple(np.random.choice(range(256), size=3))
                if not color in self.colorset:
                    unique=True
        return color

    def mask_to_box(self, mask, fr_num, img_id=None):
        """Use this function to save augmented anns, semantic, and panoptic mask for both theta=0 or theta>0
        -mask is already augmented by rotation for theta>0
        """
        self.fr_boxs = []
        self.fr_masks = []
        self.areas = []
        self.segm_ids = []

        H,W = mask.shape
        label_mask = label(mask>0)
        img_props_ind = list(np.unique(label_mask))
 
        print(f'total flower found {len(img_props_ind)}')

        self.img_mask_rgb = np.zeros(self.imgrot.shape, dtype='uint8')
        self.img_mask_semantic = np.ones(self.imgrot.shape[:2], dtype='uint8') # 1:stuff:background
        assert len(img_props_ind)<100, 'maximum possible flower annotations should be less than 100'
        for i in img_props_ind[::-1]:
            #compute instance area
            ins_mask = np.zeros(mask.shape, dtype='uint8')
            ins_mask[label_mask==i] = 255
            rle = mask_util.encode(np.asfortranarray(ins_mask))
            area = mask_util.area(rle)
            #ann_id = int(f'{img_id}{i}')
            if area>100:
                #SSL+RGR for sequential only for AppleA
                color = self.gen_unique_color()
                self.colorset.append(color)

                self.init_params['ann_id_index'][0]+=1

                
                self.img_mask_rgb[label_mask==i,0] = color[0]
                self.img_mask_rgb[label_mask==i,1] = color[1]
                self.img_mask_rgb[label_mask==i,2] = color[2]
                if i>0:
                    self.img_mask_semantic[label_mask==i] = 0 #0:thing:flower
                    self.fr_masks.append(rle)
                    #print(f'{i}: Bounding box = {mask_util.toBbox(rle)}')
                    self.areas.append(area)
                    self.segm_ids.append(int(color[0] + 256 * color[1] + 256 * 256 * color[2]))
                    x,y,w,h = mask_util.toBbox(rle)
                    self.fr_boxs.append([fr_num, i, x, y, w, h, 1, 1, 0])
                else:#background for panoptic
                    self.img_mask_semantic[label_mask==i] = 1 #1:back:flower
                    self.fr_masks.insert(0,[])
                    self.areas.insert(0, W*H - sum(self.areas))
                    self.fr_boxs.insert(0, [fr_num, i, 0,0,W,H, 1, 1, 0])
                    self.segm_ids.insert(0, int(color[0] + 256 * color[1] + 256 * 256 * color[2]))
            else:
                print(f'instance area<100')

    def remap_preds(self, mask, img_org, angle):
        """remap rotated predicted scoremap for theta>0 otherwise return original
        """
        if angle>0:
            Horg, Worg = img_org.shape[:2]
            mshape = mask.shape
            mask_rot = mask.copy()
            #mask_rot[mask_rot>0] = 255
            #print(f'org image shape {img_org.shape}')
            #print(f'rot mask shape {mask_rot.shape}')
            
            mask_rerot = imutils.rotate_bound(mask_rot, -angle)  # mask_image scaled to original image size
            Hrrot,Wrrot = mask_rerot.shape[0]//2, mask_rerot.shape[1]//2
            H,W = Horg//2, Worg//2
            mask_org = mask_rerot[Hrrot - H: Hrrot + H,Wrrot - W:Wrrot + W]
            assert (2*H,2*W)==mask_org.shape
        else:
            mask_org = mask
        assert mask_org.shape==img_org.shape[:2], f'{mask_org.shape}=={img_org.shape[:2]}'
        return mask_org

    def normalize_augment(self):
        return self.pred_scoremaps/len(self.angleSet)

    def save_scoremap(self, pred_mask, fr_num, folder='scoremap'):
        if folder=='scoremap':
            pred_mask = pred_mask*255
        else:
            pred_mask = pred_mask*255
            #pred_mask[pred_mask>0] = 255
        pred_mask=pred_mask.astype(np.uint8)
        save_path = os.path.join(self.result_path, folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pil_img = Image.fromarray(pred_mask)
        pil_img.save(f'{save_path}/{fr_num}.png')

    def save_rotation(self, pred_mask, angle, fr_num):
        pred_mask[pred_mask>0] = 255
        # pred_mask[pred_mask>=0.5] = 255 #Peach: 0.03, Pear:0.25, AppleB:0.4, AppleA: 0.45 
        # pred_mask[pred_mask<0.5] = 0
        pred_mask=pred_mask.astype(np.uint8)
        save_path = os.path.join(self.result_path, 'rotation_exp')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pil_img = Image.fromarray(pred_mask)
        pil_img.save(f'{save_path}/{fr_num}_{angle}.png')

    def get_annos_cluster_mode(self, final_scoremap, sem_mask_0=None, fr_num=None,
                               imgrot=None, angle=None, pred_score=0, verbose=False):
        """We only apply cluster mode regression? and RGR for theta=0 frames
        since sem_mask_0 isused for augmented pseudo labels. We only need rotated masks and boxs for theta>0
        """
        img_index = self.init_params['frame_index_map'][fr_num]
        img_id = int(f'{img_index}{angle}')
        #print(f'img id for anns: {img_id}')

        if angle == 0:
            final_pred = final_scoremap
            if not self.init_params['apply_rgr']:

                if fr_num in [469470000, 470160000, 472780000, 469270000, 469260000]:
                    self.save_scoremap(final_pred, fr_num, folder='normalized_map')

                final_pred[final_pred>=self.remap_score_thr] = 255 #thing: flower
                final_pred[final_pred<self.remap_score_thr] = 0 #stuff: background

            else:
                final_pred = self.rgr_refine(final_scoremap, imgrot)
            
            #self.fr_det, self.fr_mask
            self.mask_to_box(final_pred.copy(), fr_num, img_id=img_id)
            sem_mask_rot = final_pred
            sem_mask_0 = final_pred
        else:
            # use masks to remap cluster modes from theta=0 to theta= 6, 84, 90 ,....
            assert sem_mask_0 is not None
            sem_mask_rot = self.rotated_mask(sem_mask_0, angle=angle)
            #self.fr_det, self.fr_mask
            self.mask_to_box(sem_mask_rot.copy(), fr_num, img_id=img_id)
            if verbose:
                print(f'collect training examples: frame: {fr_num}, \
                #detection: {len(self.fr_det)}, angle: {angle}')
                
        return sem_mask_0, sem_mask_rot

    def get_augmented_proposals(self, fr_num):
        #get the subset of frames for a specific proces
        if fr_num not in self.semi_frames and self.init_params['database']=='flower':
            #read unlabeled images for flower
            self.im_name = os.path.join(self.init_params['unlabeled_img_dir'], '{:08d}.png'.format(fr_num))
            assert os.path.exists(self.im_name), f'{self.im_name} is not exist for {self.dataset}'
        else:
            #read labeled images: 1 to gt_fr_max
            if self.init_params['database']=='flower':
                self.im_name = os.path.join(self.init_params['train_img_dir'], '{:08d}.png'.format(fr_num))
            else:
                #for clasp we are using unlabeled frames from training set
                self.im_name = os.path.join(self.init_params['train_img_dir'], '{:08d}.png'.format(fr_num))

            assert os.path.exists(self.im_name), f'{self.im_name} is not exist for {self.database}'

        if self.init_params['database']=='flower':
            #for flower dataset select random orientations for each frames
            self.angleSet = self.get_random_angles(ranges=self.ranges, factor=self.init_params['angle_factor'])
            self.angleSet = list(set(self.angleSet))
            
        im = cv2.imread(self.im_name)
        self.rot_imgs_dict = {}
        if self.init_params['apply_rgr']:
           self.pred_scoremaps = np.zeros((self.num_sem_class, im.shape[0], im.shape[1]), dtype='float')
        else: 
            self.pred_scoremaps = np.zeros((im.shape[0], im.shape[1]), dtype='float')
        masks_0 = None
        #print(f'angle set for test-time augmentation: {self.angleSet}')
        for angle in random.sample(self.angleSet, len(self.angleSet)):
            if angle > 0:                                                      
                imgrot = imutils.rotate_bound(im, angle)
            else:
                imgrot = im
            if self.verbose:
                print(f'Image: {os.path.basename(self.im_name)}, Rotated by: {angle}')
                print(f'image shape {imgrot.shape}')

            self.rot_imgs_dict[angle] = copy.deepcopy(imgrot)

            if fr_num not in self.semi_frames:
                #return only box and thresholded binary mask: no coarse score map
                #we will do remapping and clustering using these predictions
                pred_results = self.predict_single(imgrot)
                #TODO: in current implementation, only semantic score logits are used for pseudo labels
                pred_mask = pred_results['sem_seg']#.argmax(dim=0).cpu().numpy()
                pred_mask_C = self.softmax(pred_mask)  
                

                # pred_mask = pred_mask.argmax(dim=0)
                # pred_mask = ~(pred_mask>0)
                if not self.init_params['apply_rgr']:
                    pred_mask_C = pred_mask_C.cpu().numpy() # 0: flower, 1: bck
                    pred_mask = pred_mask_C[self.flower_class_id] # 0: flower in 2 class model, 9 flower in 54 class model
                    pred_mask = np.array(pred_mask).astype('float')
                    if fr_num in [469470000, 470160000, 472780000, 469270000, 469260000] and angle==0:
                        self.save_scoremap(pred_mask, fr_num)

                    if fr_num in [467980000]:
                        self.save_rotation(pred_mask, angle, fr_num)
                    # get remapped dets and maintain the index of dets_rot
                    pred_mask_remap = self.remap_preds(pred_mask, im, angle)
                    # append detections for each orientation
                    self.pred_scoremaps+=pred_mask_remap
                else:
                    if self.pretrained:
                        pred_mask_C = torch.cat((pred_mask_C[9].unsqueeze(0), pred_mask_C[0].unsqueeze(0)))
                    pred_mask_C = pred_mask_C.cpu().numpy() # 0: flower, 1: bck
                    pred_mask_remap = [np.expand_dims(self.remap_preds(pred_mask, im, angle), axis=0) for pred_mask in pred_mask_C]                
                    self.pred_scoremaps+=np.concatenate(pred_mask_remap)

    def save_semantic_mask(self, imgname):
        """It maps all thing (flower in instance json) categories to class 0, and maps all unlabeled pixels to class 255.
        It maps all stuff categories to contiguous ids starting from 1.
        """
        #TODO: refine semantic mask using area threshold and RGR
        # #print(f'unique semantic pixels: {bin_mask}')
        # smask=np.zeros(bin_mask.shape)
        # #assign thing and stuff class (unlabeled pixels will be assigned as 255)
        # smask[bin_mask==0]=1  #assign background ((stuff class)) pixels
        # smask[bin_mask>0]=0    #assign flower (thing class) pixels

        self.img_mask_semantic=self.img_mask_semantic.astype(np.uint8)
        #print(f'semantic ids: {np.unique(self.img_mask_semantic)[::-1]}')
        #assert len(np.unique(smask))==2, f"thing (0:flower) and stuff (1:background) class ids {np.unique(smask)}"
        pil_img = Image.fromarray(self.img_mask_semantic).convert('L')
        pil_img.save(f'{self.instance_label_path}/{imgname}')

    def save_rgb_mask(self, imgname):
        """Useful to learn a panoptic model
        self.img_bin and self.img_rot is already augmended by rotation
        """
        cv2.imwrite(f'{self.panoptic_label_path}/{imgname}', self.img_mask_rgb)

    def pseudo_labels_selection(self, fr_num, final_scoremap, cluster_modes_instance=None):
        """inastance pseudo-labels: Cluster modes scores are considered 1 when RGR is used and flower class (1:flower) is the only labels
        semantic pseudo-labels: thing class (0: flower), background class (1:background)
        """
        sem_mask_0 = None
        #start loop for all unique angles and save corresponding image and detections
        # use percent GT json for semi-supervised
        if self.semi_supervised and fr_num in self.semi_frames:
            # update cluster modes: [0:fr, 1:ind, 2:x, 3:y, 4:w, 5:h, 6:score, 7:class, 8:angle] from box gt
            # and then apply augmentation
            # cluster_modes (only box) are replaced from the manual labels
            anns_fr = get_frame_anns(self.init_params['percent_clasp_gt'], fr_num)

        if len(final_scoremap)>0 or len(anns_fr) > 0:
            if self.init_params['database']=='flower':
                self.pseudo_labels_angles = random.sample(self.angleSet, len(self.angleSet)) #AppleA: 10 angles
            else: 
                self.pseudo_labels_angles = [6, 90,180, 270, 354] #[6, 84, 90, 96, 174, 180, 186, 270, 354]
            #self.pseudo_labels_angles = random.sample(self.pseudo_labels_angles, len(self.pseudo_labels_angles))

            # first angle should be zero
            if 0 not in self.pseudo_labels_angles:
                self.pseudo_labels_angles.insert(0, 0)
            else:
                self.pseudo_labels_angles.remove(0)
                self.pseudo_labels_angles.insert(0, 0)
            # theta=180 is will be used furing training
            # if 180 not in self.pseudo_labels_angles:
            #       self.pseudo_labels_angles.append(180)
            print(f'angle set for pseudo labels: {self.pseudo_labels_angles}')
            assert len(self.pseudo_labels_angles)==len(set(self.pseudo_labels_angles))

            for theta in self.pseudo_labels_angles:
                self.imgrot = self.rot_imgs_dict[theta]
                # flower: apply regression only for pseudo labels only
                if self.init_params['database'] in ['flower']:
                    # TODO: apply RGR
                    sem_mask_0, sem_mask_rot = self.get_annos_cluster_mode(final_scoremap,
                                                    sem_mask_0=sem_mask_0, fr_num=fr_num,
                                                    imgrot=self.imgrot.copy(), pred_score=0,
                                                    angle=theta, verbose=self.verbose)

                else:
                    print('{} is not available'.format(self.init_params['database']))
                    self.fr_boxs = []
                    self.fr_mask = []
                    sem_mask_rot = []

                if len(self.fr_boxs[1:])>0: # skip only background class images
                    #print(self.fr_boxs)
                    if self.verbose:
                        print(f'training examples: frame: {fr_num}, #detection: {len(self.fr_boxs)}, angle: {theta}')
                    #save image info
                    imgIdnew = 10000 * int('%06d' % fr_num) + theta
                    imgname = '{:08d}.png'.format(imgIdnew)
                    img_write_path = self.init_params['AugImgDir'] + '/' + imgname
                    cv2.imwrite(img_write_path, self.imgrot)

                    self.save_semantic_mask(imgname)
                    self.save_rgb_mask(imgname)

                    self.dataset_flower_ins = Write_ImagesInfo(self.imgrot, imgname, int(imgIdnew), self.dataset_flower_ins)
                    self.dataset_flower_pan = Write_ImagesInfo(self.imgrot, imgname, int(imgIdnew), self.dataset_flower_pan)
                    self.frame_ids.append(imgIdnew)

                    if fr_num not in self.semi_frames and theta in [0] and len(self.fr_boxs[1:])>0:
                    #    jitt_boxes, jitt_masks = self.boxes_regression(self.imgrot, np.array(self.fr_boxs[1:]), 
                    #                             jitter_boxes=True, score_thr=0.01, show_jitter=False)
                       self.vis_gt(self.imgrot, self.fr_boxs[1:], masks=self.fr_masks[1:], gt_vis_path=self.vis_path, imname=img_write_path)
                       #self.vis_gt(self.imgrot, self.fr_boxs[1:], masks=self.fr_masks[1:], gt_vis_path=self.vis_path, imname=img_write_path)


                    # save anns
                    segment_infos = []
                    for ib, box in enumerate(self.fr_boxs):
                        bboxfinal = [round(x, 2) for x in box[2:6]]
                        #segmPolys = self.fr_masks[ib]
                        mask = self.fr_masks[ib]
                        #area = mask_util.area(mask)#rle area
                        area = self.areas[ib]
                        score = 1
                        if ib>0:
                            catID = 1
                            #maintain unique anns ids in multiple process
                            annID = self.segm_ids[ib] #1000 * int('%06d' % (self.ann_id_count))
                            self.ann_ids.append(annID)

                            segmPolys = []  # mask['counts'].decode("utf-8") #[]
                            bmask = mask_util.decode(mask)
                            contours = measure.find_contours(bmask, 0.5)
                            for contour in contours:
                                contour = np.flip(contour, axis=1)
                                segmentation = contour.ravel().tolist()
                                if len(segmentation)>0:
                                    segmPolys.append(segmentation)
                            assert int(imgIdnew)== int(os.path.basename(img_write_path).split('.')[0])
                            #save annotation for for each image
                            if len(segmPolys)>0:
                                self.dataset_flower_ins = Write_AnnotationInfo(bboxfinal, segmPolys, int(imgIdnew),
                                                                    int(annID), catID, int(area), self.dataset_flower_ins,
                                                                            instance_certainty=score)
                                segment_infos = Write_Segments_info_pan(segment_infos, bboxfinal, int(annID), 
                                                                                catID, int(area), instance_certainty=score)
                        else:
                            #background class
                            catID = 0
                            annID = self.segm_ids[ib]
                            segment_infos = Write_Segments_info_pan(segment_infos, bboxfinal, int(annID), 
                                                        catID, int(area), instance_certainty=score)

                    self.dataset_flower_pan = init_annotations_info_pan(self.dataset_flower_pan, segment_infos, int(imgIdnew), imgname)

    def pseudo_labels_main(self):
        """
        Iterate over a video frames to collect the automatically generated semantic, instance, and panoptic pseudo labels
        """
        for fr_num in random.sample(self.img_subset, 10):#len(self.img_subset)
            start_time = time.time()
            # get augmented scoremaps
            self.get_augmented_proposals(fr_num)
            # normalize and refine scoremap
            if len(self.pred_scoremaps) > 0:
                try:
                    #to handle cropped frames without flower
                    assert len(self.pred_scoremaps) > 0
                    final_scoremap = self.normalize_augment()                   
                except:
                    final_scoremap = []
            else:
                final_scoremap = []

            if fr_num not in self.semi_frames:
                # use cluster modes to get pseudo-labels and
                # update coco data structure dict to populate the selected pseudo labels
                # to handle cropped frames without flower
                if len(final_scoremap) > 0:
                    self.pseudo_labels_selection(fr_num, final_scoremap)
                else:
                    print(f'no pseudo labels found at {fr_num}')
            else:
                self.pseudo_labels_selection(fr_num, final_scoremap)

            if fr_num in self.semi_frames:
                print(f"PID: {self.p_id}, Labeled Frame: {fr_num}, Execution time {time.time() - start_time} secs")
            else:
                print(f"PID: {self.p_id}, Unlabeled Frame: {fr_num}, Execution time {time.time() - start_time} secs")


def pseudo_main(q, init_params, i, process_frames,
                ann_ids, frame_ids, all_rgb, dataset_flower_ins, dataset_flower_pan):
    gpu_id = init_params['cuda_list'][i]
    init_params['process_id'] = i
    print(f'initiate process {i} using cuda {gpu_id} for total frames {len(process_frames)}')
    pseudo_labeler = Pseudo_Labels(gpu_id, init_params,  process_frames,
                                   ann_ids, frame_ids, all_rgb, dataset_flower_ins, dataset_flower_pan,
                                   p_id=i,verbose=init_params['verbose'])
    #pseudo_labeler.init_detector()
    pseudo_labeler.pseudo_labels_main()
    #, 'train_data':dataset_clasp, 'all_scores':all_scores
    print(f'PID: {i} finished assigned frames')
    q.put({'process_finished': True})

def split_frames(num_processes, all_frames):
    max_limit = len(all_frames)//num_processes
    assert max_limit!=1
    lengths_frame = [max_limit] * num_processes
    lengths_frame[-1] += abs(sum([max_limit] * num_processes) - len(all_frames))
    all_frames_iter = iter(all_frames)
    return [list(islice(all_frames_iter, elem)) for elem in lengths_frame]

def gen_all_rgb(mp_dict, total_frames=3000, total_angles=20, ins_per_image=10):
    """useful in multiprocess
    """
    colors = {}
    colors['RGB'] = []
    #for color in itertools.product(np.arange(128, 256), repeat=3):
    for R in np.arange(128, 256):
        for G in np.arange(128, 256):
            for B in  np.arange(0, 48):
                colors['RGB'].append((R, G, B))
                #print(colors['RGB'])
    return colors

def gen_unique_colors(num_frames=3000, rot_angles=10, max_targets_per_frame=50):
    total_colors = num_frames*rot_angles*max_targets_per_frame
    colorset = set()
    while len(colorset) <= total_colors:
        color = tuple(np.random.choice(range(256), size=3))
        if not color in colorset:
            colorset.add(color)
    return colorset

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--data_set', dest='data_set', required=True,
        help='dataset to use: AppleA or Peach or Pear or AppleB', default='AppeleA')
    
    parser.add_argument(
        '--database', dest='database', required=True,
        help='Database to use: clasp1 or clasp2', default='clasp2')

    parser.add_argument(
        '--label_percent',
        help='percent of manual annotations used in semi-SL',
        default=1, type=int)
    
    parser.add_argument(
        '--CV',
        help='Cross Validation in SSL',
        default=1, type=int)
    
    parser.add_argument(
        '--ssl_iter',
        help='SSL iteration index to make sure the pretrained model is loaded properly',
        default=0, type=int)
    
    parser.add_argument(
        '--angle_factor',
        help='factor to generate stratified rotation angle samples',
        default=4, type=int)
    
    parser.add_argument(
        '--number_gpus',
        help='#GPUS in SSL',
        default=2, type=int)
    
    parser.add_argument(
        '--pretrained',
        help='#GPUS in SSL',
        default=0, type=int)

    parser.add_argument('--model_type', type=str, default='SSL')

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

    return parser.parse_args()

if __name__ == '__main__':
    
    mp.set_start_method('forkserver', force=True)

    args = parse_args()
    print('Called with args:')
    print(args)
    # start multiprocess for multi-camera pseudo label generation
    num_gpus=args.number_gpus
    model_type = args.model_type
    exp = args.ssl_iter
    
    init_params = {}
    init_params['database'] = args.database
    init_params['data'] = f'{args.database}_2021'
    init_params['storage_dir'] = coderoot

    init_params['regress_aug_prop'] = True
    init_params['semi_supervised'] = True #SSL
    init_params['apply_cluster_mode'] = True
    init_params['verbose'] = False

    if args.database in ['flower']:
        if model_type in ['SSL']:
            init_params['apply_rgr'] = 0
        else:
            init_params['apply_rgr'] = 1
            
        init_params['CV'] = args.CV
        init_params['angle_factor'] = args.angle_factor
        # use 0.5++
        if num_gpus==2:
            init_params['cuda_list'] = [1,1]
        if num_gpus==4:
           init_params['cuda_list'] = [0,0,1,1,2,2,3,3]
        init_params['class_list'] = [1]
        init_params['nms_thr'] = 0.6
        init_params['pred_score'] = 0.005
        init_params['num_sem_class'] = 2 #0:thing:flower, 1:stuff:background

        init_params['iter_i'] = 1
        init_params['pretrained'] = args.pretrained
        init_params['default_predictor'] = 0
        init_params['test_aug'] = 0
        init_params['angle_ranges'] = [[1, 24], [78, 96], [168, 192], [258, 276], [342, 354]]
        init_params['combine_panoptic'] = 0
        #for semantic soft prediction
        init_params['num_class'] = 2
        init_params['class_ids'] = {'flower':0, 'background':1}
        init_params['data_already_registered'] = False
        #use 0.4-0.5 or nms to avoid noisy clustering
        init_params['cluster_score_thr'] = [0.4, 0.4]
        # currently unused: useful to discard noisy samples from the augmented proposals
        init_params['sem_thr'] = {'AppleA':{1:0.32, 2:0.56, 3:0.5, 4:0.48}, 
                                  'AppleB':{1:0.4, 2:0.52, 3:0.42},
                                  'Peach':{1:0.1, 2:0.16, 3:0.6, 4:0.5, 5:0.58, 6:0.52},
                                  'Pear':{1:0.32, 2:0.22, 3:0.48}
                                  }
        
        init_params['regress_pred_score'] = 0.4
        init_params['remap_score_thr'] = init_params['sem_thr'][args.data_set][args.CV]
        
    
    init_params['save_data'] = True
    init_params['angleSet'] = [0, 6, 12, 78, 84, 90, 96, 102, 168, 174,
                               180, 186, 192, 258, 264, 270, 276, 342, 348, 354]
    init_params['angle_ranges'] = [[1, 24], [78, 96], [168, 192], [258, 276], [342, 354]]

    #get all required dirs
    init_params, all_frames = get_all_dirs(args, exp, init_params, coderoot, model_type)

    #multiple process for pseudo-labels generation
    manager = mp.Manager()
    ann_ids = manager.list()
    #all_rgb = gen_all_rgb(manager.dict())
    all_rgb = list(gen_unique_colors(num_frames=20, rot_angles=10, max_targets_per_frame=50))
    init_params['ann_id_index'] = manager.list()
    init_params['ann_id_index'].append(0)

    print('total rgb colors: {}'.format(len(all_rgb)))

    frame_ids = manager.list()
    init_params['colorset'] = manager.list()
    # initiate dataset_clasp dictionary using global mp dicts to update train data
    dataset_flower_ins = define_dataset_dictionary_mp(manager, database=args.database, mode='instance')
    dataset_flower_pan = define_dataset_dictionary_mp(manager, database=args.database, mode='panoptic')
    #dataset_clasp = define_dataset_dictionary_mp(manager, database=args.database)
    print(dataset_flower_ins, dataset_flower_pan)

    #set of gpus
    num_gpus = torch.cuda.device_count()

    #split all_frames to all processes
    num_processes = len(init_params['cuda_list'])
    all_frames = random.sample(all_frames, len(all_frames))
    init_params['frame_index_map'] = {frame:index+1 for index, frame in enumerate(all_frames)}
    splitted_frames = split_frames(num_processes, all_frames)
    assert len(set(all_frames))==sum([len(set(grp)) for grp in splitted_frames])==len(all_frames)

    #list of processes
    q=list(range(num_processes))
    start_time = time.time()
    #assign processes to each gpu
    p = {}
    for i in q:
        q[i] = mp.Queue()
        # Pass GPU number through q
        p[i] = mp.Process(target=pseudo_main,
                       args=(q[i], init_params, i, splitted_frames[i],
                             ann_ids, frame_ids, all_rgb, dataset_flower_ins, dataset_flower_pan)
                       )
        p[i].start()

    #main loop to save psudo labels JSON
    while True:
        check_terminate = [q[i].get()['process_finished'] for i in range(len(q))]
        if all(check_terminate):
            print('all queue status {}'.format(check_terminate))

            if init_params['save_data']:
                dataset_flower_ins = Proxy2RealObj(dataset_flower_ins, database=args.database)
                dataset_flower_pan = Proxy2RealObj(dataset_flower_pan, database=args.database)
                Write_To_Json(init_params['instance_json'], dataset_flower_ins)
                Write_To_Json(init_params['panoptic_json'], dataset_flower_pan)
                print(f"total time for {args.database}: {time.time() - start_time} sec")
                #TODO: test uniqueness of all annotation ids
                # image ids and the anns ids should be unique for training dataset
                assert len(frame_ids)==len(set(frame_ids)), f'frame ids {len(frame_ids)}, set of frame ids {len(set(frame_ids))}'
                assert len(ann_ids)==len(set(ann_ids)), f'{len(ann_ids)} == {len(set(ann_ids))}'
                # Ctrl+/
            # terminate each sub-processes
            for i in list(range(num_processes)):
                p[i].join()
                p[i].terminate() 
            break
        else:
            continue
