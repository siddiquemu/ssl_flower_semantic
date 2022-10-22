#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:29:47 2021

@author: walden
"""

import numpy as np
import cv2 as cv
import os, os.path
import json
import pandas
import PIL
from PIL import Image, ImageDraw


#%% function for generating a unique color given the set of colors previously
## used. Gets called by the image creation function. The colorset should be a 
## global dictionary, i.e. all images append to the same one 
def gen_unique_color(colorset):
    unique=False
    while not unique:
            color = tuple(np.random.choice(range(256), size=3))
            if not(color in colorset):
                unique=True
    return color

#%% function for returning the outer and inner segmentation of am image,
# as well dictionary containing RLE,h,w,bbox,area,ID for each segment

def filter_contours(img):
    h,w=img.shape
    contours_all_buf, hier_buf= cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)   ##internal and external contours
    
    contours_all=[]
    hier=[]
    for i in range(len(contours_all_buf)):
        if len(contours_all_buf[i])>=6:
            contours_all.append(contours_all_buf[i])
            hier.append(hier_buf[0,i,:])
            
    hier=[hier]
    segmentation_all = cnt2seg(contours_all)
    
    bmask=PIL.Image.new(mode="RGB", size=(w,h))
    bmaskdraw=ImageDraw.Draw(bmask)##enables drawing in the mask
    
    contours_inner=[]
    contours_outer=[]
    
    for i in range(len(segmentation_all)):
        exists_parent=(hier[0][i][3]!=-1)
        parent_seg=segmentation_all[i]
        if not(exists_parent) and  (cv.contourArea(contours_all[i])>25):
            contours_outer.append(contours_all[i])
            children=[]
            for j in range(len(segmentation_all)):
                parent=hier[0][j][3]
                child=segmentation_all[j]
                if parent==i:
                    children.append(child)
            bmaskdraw.polygon(parent_seg, fill=(255,255,255), outline=(255,255,255))
            for child_seg in children:
                bmaskdraw.polygon(child_seg, fill=(0,0,0), outline=(0,0,0))
            bmask=np.asfortranarray(np.array(bmask))
        else:
            contours_inner.append(contours_all[i])
        
    return contours_inner, contours_outer, h, w

#%% function to turn the inner and outer segmentation into color mask. Also
# returns lists with segIDs, bboxes, areas. Indices of these lists line up so
# they can be for-looped through later

def rgb_mask_gen(colorsdict,inners,outers,h,w):
    
    segIDs=[]
    bboxes=[]
    areas=[]
    
    outers_cnt=outers
    outers_seg=cnt2seg(outers)

    #include rgb id for background stuff class



    color = gen_unique_color(colorsdict)
    rgbmask = PIL.Image.new(mode="RGB", size=(w, h), color=color)
    mask_array = np.array(rgbmask)
    img_area = w * h
    #mask_array[:, :, 0][mask_array[: ,: , 0] == 0] = color[0]
    #mask_array[:, :, 1][mask_array[:, :, 1] == 0] = color[1]
    #mask_array[:, :, 2][mask_array[:, :, 2] == 0] = color[2]
    segIDs.append(int(color[0] + 256 * color[1] + 256 * 256 * color[2]))
    areas.append(img_area)
    bboxes.append([0, 0, w, h])
    rgbmask = Image.fromarray(mask_array.astype('uint8'), 'RGB')
    rgbmaskdraw=ImageDraw.Draw(rgbmask)##enables drawing in the mask
    
    for i in range(len(outers)):
        cnt=outers_cnt[i]
        seg=outers_seg[i]
        area_n = cv.contourArea(cnt)
        
        #no need to perform area thresholding here, I do it earlier
        
        color=gen_unique_color(colorsdict)
        colorsdict.add(color)
        rgbmaskdraw.polygon(seg, fill=color, outline=color)
    
        segID_n=int(color[0] + 256 * color[1] + 256 * 256 * color[2])
        x_n,y_n,w_n,h_n=cv.boundingRect(cnt)
        segIDs.append(segID_n)
        areas.append(area_n)
        bboxes.append([x_n,y_n,w_n,h_n])
        #rgbmaskdraw.polygon(seg, fill=(0, 0, 0), outline=(0, 0, 0))

    #inners=cnt2seg(inners)
    #for seg in inners:
        #rgbmaskdraw.polygon(seg, fill=(0,0,0), outline=(0,0,0))

    return colorsdict, rgbmask, segIDs, bboxes, areas


#%%function for converting contours format to segmentation format

def cnt2seg(contours):
    segmentation = [x.flatten() for x in contours]
    segmentation= [[float(y) for y in x] for x in segmentation if len(x) >=6]
    return segmentation

#%%function for generation [segment_info] portion of the panoptic annotation
def pan_ann_gen(outers,segIDs,imgID,bboxes,areas, img_area=None):#RLEs is not a necessary argument
    segment_info=[]
    for i in range(len(segIDs)):
        area=areas[i]
        bbox=bboxes[i]
        segID=segIDs[i]
        if area==img_area:
            print('found panoptic background category of area {}'.format(area))
            catID=2
            new_seg_pan={
                'id': segID,
                'category_id': catID,
                'area': 2*img_area-sum(areas),
                'bbox': bbox,
                'iscrowd': 0
                }
            segment_info.append(new_seg_pan)
        else:
            catID=1
            new_seg_pan={
                'id': segID,
                'category_id': catID,
                'area': area,
                'bbox': bbox,
                'iscrowd': 0
                }
            segment_info.append(new_seg_pan)
    
    filename=str(imgID)+'.png'
    pan_ann={
        'segments_info': segment_info,
        'image_id': imgID,
        'file_name': filename
        }
    
    return pan_ann

#%%function for generating instance annotations
def inst_ann_gen(outers,segIDs,imgID,bboxes,areas):
    inst_ann=[]
    outers=cnt2seg(outers)
    for i in range(len(outers)):
# =============================================================================
#         segmentation=outers[i]
#         area=areas[i]       #[i+1]
#         bbox=bboxes[i]      #[i+1]
#         segID=segIDs[i]     #[i+1]
# =============================================================================
        segmentation=outers[i]
        area=areas[i+1]     #first item in the list is the background
        bbox=bboxes[i+1]
        segID=segIDs[i+1]
        catID=1
        new_seg_inst={
            'segmentation':[segmentation],
            'area': area,
            'iscrowd': 0,
            'image_id': imgID,
            'bbox': bbox,
            'category_id': catID,
            'id': segID
            }
        inst_ann.append(new_seg_inst)
    return inst_ann

#%% funtion for extracting IDs (int) and filenames (strings) from .csv files
def gen_lists(path):
    
    flowerfile=path+'/flower.csv'
    maskfile=path+'/mask.csv'
    
    flowerfile=pandas.read_csv(flowerfile,header=None)
    flowerfile.columns=['ID','filename']
    maskfile=pandas.read_csv(maskfile,header=None)
    maskfile.columns=['ID','filename']
    
    flower_filenames=list(flowerfile['filename'])
    mask_filenames=list(maskfile['filename'])
    
    assert(all(flowerfile['ID']==maskfile['ID']))
    IDs=list(flowerfile['ID'])
 
    return IDs, flower_filenames, mask_filenames

#%% function for generating categories list of dictionaries

def define_categories_dictionary():
    categories=[
        {
        'id': 1,
        'name': 'flower',
        'supercategory': 'flower',
        'isthing': 1,
        'color': []
        },
        {
        'id': 2,
        'name': 'background',
        'supercategory': 'background',
        'isthing':0,
        'color':[]
        }]
    return categories

#%% main method

det2 = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/Detectron2'
base_dir = '/home/walden/Desktop/appleData/trainTestSplit/test/dataFormattedProperly/splitImages4x3/'
bmask_dir=base_dir+'bMasksSplit/'
flower_dir=base_dir+'flowersSplit/'
rgbmask_dir=base_dir+'rgbMasksSplit/'

imgIDs,flower_filenames,mask_filenames=gen_lists(base_dir)

pan_anns=[]
inst_anns=[]
imgs=[]
colorsdict=set()

for i in range(len(imgIDs)):
    print('image id: {}'.format(imgIDs[i]))
    img=cv.imread(bmask_dir+mask_filenames[i],0)
    img_area = img.shape[0] * img.shape[1]
    imgID=imgIDs[i]
    
    inners, outers, h, w=filter_contours(img)
    colorsdict, rgbmask, segIDs, bboxes, areas= rgb_mask_gen(colorsdict, inners, outers, h, w)
    rgbmask.save(rgbmask_dir+mask_filenames[i])
    
    pan_ann=pan_ann_gen(outers, segIDs, imgID, bboxes, areas, img_area=img_area)
    pan_anns.append(pan_ann)
    
    inst_ann=inst_ann_gen(outers, segIDs, imgID, bboxes, areas)
    inst_anns.extend(inst_ann)
        
    
    img_n={
        'license': None,
        'file_name': flower_filenames[i],
        'coco_url': None,
        'height': h, #height
        'width': w, #width
        'date_captured': None,
        'flickr_url': None,
        'id': imgID
        }
    imgs.append(img_n)


cats=define_categories_dictionary()

panopticjson={}
panopticjson['images']=imgs
panopticjson['annotations']=pan_anns
panopticjson['categories']=cats

s=json.dumps(panopticjson)
with open(base_dir+'appleA_panoptic_split4x3_test.json','w') as f:
    f.write(s)

instancejson={}
instancejson['images']=imgs
instancejson['annotations']=inst_anns
instancejson['categories']=cats

s=json.dumps(instancejson)
with open(base_dir+'appleA_instance_split4x3_test.json','w') as f:
    f.write(s)

#python /home/siddique/miniconda3/envs/det2/lib/python3.7/site-packages/panopticapi/evaluation.py --gt_json_file /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/Detectron2/flower_dataset/AppleA_Panoptic/DataFormattedProperly/SplitImages4x3/RLE_and_polygon/polygon/appleA_panoptic_split4x3_polygon.json --pred_json_file /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/Detectron2/flower_dataset/AppleA_Panoptic/DataFormattedProperly/SplitImages4x3/RLE_and_polygon/polygon/appleA_panoptic_split4x3_polygon.json --gt_folder /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/Detectron2/flower_dataset/AppleA_Panoptic/DataFormattedProperly/SplitImages4x3/RLE_and_polygon/polygon/rgbMasksSplit --pred_folder /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/Detectron2/flower_dataset/AppleA_Panoptic/DataFormattedProperly/SplitImages4x3/RLE_and_polygon/polygon/rgbMasksSplit