import copy
import glob
import os
import sys

coderoot = os.path.dirname(os.path.realpath(__file__)).split('utils')[0]
print(f'coderoot:{coderoot}')
sys.path.insert(0, f"{coderoot}")
sys.path.insert(0, f"{coderoot}rgr-public/Python")
from runRGR import RGR
import cv2
import time
import pdb

import matplotlib.pyplot as plt
import numpy as np
from src.panoptic_flower_model import panoptic_fpn_flower
from box_jitter import aug_box_jitter
from utils import get_data_dirs
import random
from PIL import ImageColor
from skimage.morphology import label
from skimage.measure import regionprops
import pycocotools.mask as mask_util
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy.io as sio
from tools.evaluate_mask import evaluate_mask
from tabulate import tabulate
import imutils
from tools.flower_counting import flower_counter
import pandas as pd


class sliding_windows(object):
    def __init__(self, img, step_size=None, window_size=None, gpu_id=0, init_params=None, detector=None):
        self.img=img
        self.step_size=step_size # step_size = (w_step, h_step)
        self.window_size=window_size #window_size = (w, h)
        self.gpu_id = gpu_id
        self.ranges = init_params['angle_ranges']
        self.angleSet = self.get_random_angles(ranges=self.ranges, factor=init_params['rotation_factor'])
        if init_params['test_aug']:
            print(f'angle set for test-time augmentation: {self.angleSet}')
        self.angleSet = list(set(self.angleSet))
        self.flower_detector = detector
        self.softmax = nn.Softmax(dim=0)

    @staticmethod
    def pad_img(img, step_size):
        # cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
        # top: It is the border width in number of pixels in top direction. 
        # bottom: It is the border width in number of pixels in bottom direction. 
        # left: It is the border width in number of pixels in left direction. 
        # right: It is the border width in number of pixels in right direction.
        return cv2.copyMakeBorder(img, step_size[1], step_size[1],
                                      step_size[0], step_size[0],
                                      cv2.BORDER_CONSTANT, None, value = 0)
    @staticmethod
    def get_random_angles(ranges=None, factor=None):
        """
        Apply stratified sampling for the rotation set
        """
        angleSet = [0, 180]
        for intvl in ranges:
            angleSet += random.sample(range(intvl[0], intvl[1]), factor)
        return angleSet[0:-2]

    @staticmethod
    def box_mask_overlay(ref_box, contours, ax, im_h, im_w, 
                        score_text, color_mask, box_color, show_box=False, 
                        show_cat=False, line_width=2):
        box_coord = ref_box[0:4].astype(int)
        # why only consider bbox boundary for mask???? box can miss part of the object
        # overlay mask on image frame
        x_0 = max(box_coord[0], 0)
        x_1 = min(box_coord[0] + box_coord[2] , im_w)
        y_0 = max(box_coord[1], 0)
        y_1 = min(box_coord[1] + box_coord[3] , im_h)
        #pdb.set_trace()
        if contours is not None:

            for c in contours:
                polygon = Polygon(
                    c.reshape((-1,2)),
                    fill=False, facecolor=color_mask,
                    edgecolor=color_mask, linewidth=line_width,
                    alpha=1.0)
                ax.add_patch(polygon)
        # show box
        if show_box:
            ax.add_patch(
                plt.Rectangle((x_0, y_0),
                            box_coord[2],
                            box_coord[3],
                            fill=False, edgecolor=box_color,
                            linewidth=2, alpha=0.5))
        if show_cat:
            ax.text(
                int(box_coord[0]), int(box_coord[1]),
                score_text,
                fontsize=16,
                family='serif',
                #bbox=dict(facecolor=box_color,alpha=0.5, pad=0, edgecolor='none'),#
                color='red')
        return ax

    def sliding_window(self):
        # slide a window across the image
        #before padding
        #H, W = self.img.shape[:2]
        #print(f'before padding: {self.img.shape}')
        #pdb.set_trace()
        #self.img = self.pad_img(self.img, self.step_size)
        for y in range(0, self.img.shape[0]-self.step_size[1], self.step_size[1]):
            for x in range(0, self.img.shape[1]-self.step_size[0], self.step_size[0]):
                # yield the current window
                yield (x, y, x + self.window_size[0], y + self.window_size[1], self.img[y:y + self.window_size[1], x:x + self.window_size[0]])

    def remove_pad(self, pad_img, img_shape):
        H, W = pad_img.shape[:2]
        h, w = img_shape[:2]
        return pad_img[H//2-h//2: H//2+h//2, W//2-w//2: W//2+w//2]
    
    def remove_pad_scores(self, pad_img, img_shape):
        return pad_img[:, self.step_size[1]-1:img_shape[0]+self.step_size[1]-1,
               self.step_size[0]-1:img_shape[1]+self.step_size[0]-1]

    def get_padded_mask(self):
        img = self.pad_img(self.img, self.step_size).copy()
        return np.zeros((img.shape[0], img.shape[1]), dtype='float')
    
    def get_padded_gt(self, img):
        return cv2.copyMakeBorder(img, self.step_size[1], self.step_size[1],
                                      self.step_size[0], self.step_size[0],
                                      cv2.BORDER_CONSTANT, None, value = 0)

    def show_input_wndows(self, x,y):
        clone = self.img.copy()
        clone = cv2.rectangle(clone, (x, y), (x + self.window_size[0], y + self.window_size[1]), (0, 255, 0), 2)
        final_img = cv2.resize(clone, (self.img.shape[1]//4, self.img.shape[0]//3))
        cv2.imshow('image',final_img)
        cv2.waitKey(1)
        time.sleep(2)

    def print_w_stats(self):
        print('window_size: {}'.format(self.window_size))
        print('step size: {}'.format(self.step_size))
        print('image shape: {}'.format(self.img.shape))

    def save_final_pred(self, save_path, final_pred):
        #TODO: Why saved binary mask has more than two unique values?
        final_pred[final_pred>=0.1] = 255
        final_pred[final_pred < 0.1] = 0
        print('final pred unique values: {}'.format(np.unique(final_pred)))
        cv2.imwrite(save_path, final_pred)

    def get_random_color(self, num_of_colors):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(num_of_colors)]
        return color

    def vis_contours(self, img_bin, im, ax, color_mask='blue', line_width=2):
        img_mask_labels = label(img_bin,connectivity=2)
        img_props_ind = list(np.unique(img_mask_labels))
        img_props_ind.remove(0)
        print(f'total flower found {len(img_props_ind)}')
        colors = self.get_random_color(len(np.unique(img_mask_labels)))
        #pdb.set_trace()
        for i in img_props_ind:
            ins_mask = np.zeros(img_bin.shape, dtype='uint8')
            ins_mask[img_mask_labels==i] = 255
            rle = mask_util.encode(np.asfortranarray(ins_mask))
 
            # print(f'{i+1}: Centroid = {i.centroid}')
            #print(f'{i}: Bounding box = {mask_util.toBbox(rle)}')
            #box = mask_util.toBbox(rle)
            #CV2 3.x
            contours,_= cv2.findContours(ins_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #pdb.set_trace()
            if int(cv2.__version__[0])==3:
                cnt_ind = 1
            elif int(cv2.__version__[0])==4:
                cnt_ind=0
            else:
                print(f'supported OpenCV version: 3, 4')

            segmentation = []
            for contour in contours:
                # Valid polygons have >= 6 coordinates (3 points)
                if contour.size >= 6:
                    segmentation.append(contour.flatten().tolist())
            if len(segmentation)>0:
                RLEs = mask_util.frPyObjects(segmentation, ins_mask.shape[0], ins_mask.shape[1])
                RLE = mask_util.merge(RLEs)
                # RLE = mask_util.encode(np.asfortranarray(mask))
                area = mask_util.area(RLE)
                [x,y,w,h] = cv2.boundingRect(ins_mask)
                box = np.array([x,y,w,h])

                if mask_util.area(rle)>400:
                    #color_mask = np.array([1, 0, 0], dtype='float32')  # green
                    box_color = colors[i]
                    score_text = 'flower'
                    ax = self.box_mask_overlay(box, contours, ax, im.shape[0], im.shape[1],
                                            score_text, color_mask, box_color, line_width=line_width)
        return ax

    def overlay_masks(self, img_bin=None, im=None, out_dir=None, imgname=None,
                    dpi=200, box_alpha=0.2, gt=None, evaluation=None, nums=None, show_class=False, show_gt_contours=False):
        """Visual debugging of detections."""
        im = im[:, :, ::-1]  # BGR -> RGB for visualization
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        vis_dir = os.path.join(out_dir, os.path.basename(imgname).split('.')[0]+'.png')

        fig = plt.figure(frameon=False)
        fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(im)

        if img_bin is not None:
            ax = self.vis_contours(img_bin, im, ax, color_mask='blue', line_width=4)
        if show_gt_contours:
            gt[gt>0]=255
            #show instance version of semantic gt
            
            ax = self.vis_contours(gt, im, ax, color_mask='red', line_width=2)
            #show actual gt
            #plt.imshow(gt, 'jet', interpolation='none', alpha=0.5)
        
            
            #R, G, B = ImageColor.getcolor(color[id], "RGB")
        if evaluation:
            eval_text = f'{evaluation[0]}, {evaluation[1]}, {evaluation[2]}, {evaluation[3]}, {evaluation[4]}, {evaluation[5]}'
            ax.text(
                int(100), int(100),
                eval_text,
                fontsize=16,
                family='serif',
                bbox=dict(facecolor='white',alpha=0.8, pad=0, edgecolor='none'),#
                color='red') 
        if nums:
            eval_text = f'PN: {nums[0]}, TP: {nums[1]}, FN: {nums[2]}, FP: {nums[3]}'
            ax.text(
                int(100), int(200),
                eval_text,
                fontsize=16,
                family='serif',
                bbox=dict(facecolor='white',alpha=0.8, pad=0, edgecolor='none'),#
                color='red') 
        fig.savefig(vis_dir, dpi=dpi)
        plt.close('all')

    def remap_preds(self, mask, img_org, angle):
        """remap rotated predicted scoremap tensor
        """
        Horg, Worg = img_org.shape[:2]
        mask_rot = mask
        
        mask_rerot = [imutils.rotate_bound(mask, -angle) for mask in mask_rot]  # mask_image scaled to original image size
        mask_rerot = torch.stack(tuple(torch.tensor(mask_rerot).to(self.gpu_id)), dim=0)
        
        Hrrot,Wrrot = mask_rerot.shape[1]//2, mask_rerot.shape[2]//2
        H,W = Horg//2, Worg//2
        mask_org = mask_rerot[:, Hrrot - H: Hrrot + H,Wrrot - W:Wrrot + W]
        assert (2*H,2*W)==mask_org.shape[1::]
        # mask_org[mask_rot>0] = 255

        return mask_org

    def test_aug(self, im):
        pred_scoremaps = torch.zeros((2, im.shape[0], im.shape[1]), dtype=torch.float).to(self.gpu_id)
        
        assert 0 in self.angleSet
        for angle in random.sample(self.angleSet, len(self.angleSet)):
            if angle > 0:                                                      
                imgrot = imutils.rotate_bound(im, angle)
            else:
                imgrot = im
            pred_results = self.flower_detector.predict_single(imgrot)
            pred_mask = pred_results['sem_seg']#.argmax(dim=0).cpu().numpy()
            pred_mask = self.softmax(pred_mask)
            # get remapped dets and maintain the index of dets_rot
            if angle>0:
                pred_mask = pred_mask
                pred_mask = pred_mask.cpu().numpy() # 0: flower, 1: bck
                pred_mask = np.array(pred_mask).astype('float')
                mask_remap = self.remap_preds(pred_mask, im, angle)
            else:
                mask_remap = pred_mask
                pred_results_0 = pred_results

            pred_scoremaps+=mask_remap
        pred_scoremaps/=len(self.angleSet)
        return pred_scoremaps, pred_results_0

    def save_scores(self, scores_pred, out_path, imname, numCls=2):
        """save score map (CxHxW); C is for number of classes
        """
        basename = os.path.basename(imname).split('.')[0] + '.mat'
        path = f'{out_path}/panopt_scores'
        if not os.path.exists(path):
            os.makedirs(path)
        sio.savemat(f'{path}/{basename}', {'scoreMap_M':scores_pred})

    def apply_area_filter(self, bin_mask, area_thr=400):
        label_mask = label(bin_mask>0)
        img_props_ind = list(np.unique(label_mask))
        img_props_ind.remove(0)
        print(f'total flower found {len(img_props_ind)}')
        filtered_mask = np.zeros(label_mask.shape, dtype='uint8')
        for i in img_props_ind:
            #compute instance area
            ins_mask = np.zeros(label_mask.shape, dtype='uint8')
            ins_mask[label_mask==i] = 1
            rle = mask_util.encode(np.asfortranarray(ins_mask))
            area = mask_util.area(rle)
            if area>area_thr:
                filtered_mask[label_mask==i] = 1
            else:
                filtered_mask[label_mask==i] = 0
        return filtered_mask
    
    def apply_regression(self, pred, img, seg_score_thr=0.32, box_score_thr=0.0, 
    area_thr=50, jitter_size=20, jitter_factor=0.1, jitter_boxes=False):
        """filtered by box size=100, score=0.4
        jitter_boxes: jitter_times=10, jitter_scale=0.06, reg_boxes_score=0.02
        """
        #regressed final mask
        final_scoremap = np.zeros(pred[0].shape, dtype='float')
        #proposals from score map
        pred_reg = pred[0].cpu().numpy().copy()
        pred_reg[pred_reg>=seg_score_thr] = 255
        pred_reg[pred_reg<seg_score_thr] = 0

        proposals = []
        label_mask = label(pred_reg>0)
        img_props_ind = list(np.unique(label_mask))
        img_props_ind.remove(0)
        #print(f'total flower found {len(img_props_ind)}')

        if len(img_props_ind)>0:
            for i in img_props_ind:
                ins_mask = np.zeros(label_mask.shape, dtype='uint8')
                ins_mask[label_mask==i] = 1
                rle = mask_util.encode(np.asfortranarray(ins_mask))
                area = mask_util.area(rle)
                if area>area_thr:
                    proposals.append(mask_util.toBbox(rle))

            if len(proposals)>0:
                proposals = np.array(proposals)
                proposals = torch.tensor(proposals, device=self.gpu_id)
                if jitter_boxes:
                    aug_proposals = torch.reshape(proposals.clone(), (len(proposals),1,4))
                    aug_proposals = aug_box_jitter(aug_proposals, times=jitter_size, frac=jitter_factor) # size: (len(fr_boxes),10,1,4)
                    aug_proposals = torch.stack(aug_proposals, dim=0)
                    aug_proposals = torch.reshape(aug_proposals, (len(proposals)*jitter_size, 4))
                    

                pred_results = self.flower_detector.predict_single(img, pre_proposals=proposals, 
                                                                return_coarse=True, apply_jitter=jitter_boxes)
                scores = pred_results.scores.cpu().numpy()
                keep = scores>box_score_thr
                boxs = pred_results.pred_boxes.tensor.cpu().numpy()
                boxs = boxs[keep]
                # boxs[:,2:4] -= boxs[:,0:2]
                boxs = boxs.astype('int')
                masks = pred_results.pred_masks.cpu().numpy()
                masks = masks[keep]

                for i, bb in enumerate(boxs):
                    x, y = bb[0], bb[1]
                    w = max(0, bb[2]-x)
                    h = max(0, bb[3]-y)
                    if w>0 and h>0:
                        score_obj = cv2.resize(masks[i,0,:,:],(w,h))
                        # print(img.shape, final_scoremap.shape)
                        final_scoremap[bb[1]:bb[3], bb[0]:bb[2]] = score_obj
                        #final_scoremap +=np.array(masks[i], dtype='uint8')

        final_scoremap = torch.tensor(final_scoremap).to(self.gpu_id)
        #dummy bck
        final_scoremap = torch.stack((final_scoremap, final_scoremap))
        return final_scoremap
    
    def rgr_refine(self, pred, img):
        #call RGR
        print(f'refining using RGR...')
        pred = np.transpose(pred, (1,2,0))
        soft_mask = np.zeros(pred.shape, dtype='float')
        soft_mask[:,:,0] = pred[:,:,1]
        soft_mask[:,:,1] = pred[:,:,0]
        #
        # soft_mask[:, :, 1][soft_mask[:, :, 1]<0.16] = 0.005
        # soft_mask[:, :, 0] = 1 - soft_mask[:, :, 1]
        assert img.shape[:2]==pred.shape[:2]
        
        # fixed parameters
        numSets = 10    # number of seeds sets (samplings)
        cellSize = 100   # average spacing between samples

        ## RGR parameters
        # thresholds
        tau0 = 0.005  # original CNN threshold 0.5
        tauF = 0.1  # high confidence foreground 0.8
        tauB = 0.0001     # high confidence background
        m = 0.1
        im_color, finalMask = RGR(img, soft_mask, m, numSets, cellSize, tau0, tauF, tauB)
        finalMask = cv2.cvtColor(finalMask, cv2.COLOR_RGB2GRAY)
        finalMask[finalMask>0]=1
        return finalMask

    def instances_to_semantic(self, im, pred_results, box_score_thr=0.5):
        final_scoremap = np.zeros(im.shape[:2], dtype='float')

        scores = pred_results.scores.cpu().numpy()
        keep = scores>=box_score_thr
        boxs = pred_results.pred_boxes.tensor.cpu().numpy()
        boxs = boxs[keep]
        # boxs[:,2:4] -= boxs[:,0:2]
        boxs = boxs.astype('int')
        masks = pred_results.pred_masks.cpu().numpy()
        masks = masks[keep]

        for i, bb in enumerate(boxs):
            x, y = bb[0], bb[1]
            w = max(0, bb[2]-x)
            h = max(0, bb[3]-y)
            if w*h>0:
                score_obj = cv2.resize(masks[i,0,:,:],(w,h))
                final_scoremap[bb[1]:bb[3], bb[0]:bb[2]] = score_obj
                #final_scoremap+=np.array(masks[i], dtype='uint8')

        final_scoremap = torch.tensor(final_scoremap).to(self.gpu_id)
        #dummy bck
        final_scoremap = torch.stack((final_scoremap, final_scoremap))
        return final_scoremap
    
    def get_gt_mask(self, fr, data_type, dataroot):

        gt_path = f'{dataroot}/raw_data/labels/{data_type}/gt_frames'
        if data_type=='Pear':
            gt_m = cv2.imread(gt_path + '/{:03d}.png'.format(int(fr)))
        else:
            gt_m = cv2.imread(gt_path + '/{:03d}.png'.format(int(fr)))

        gt_m = cv2.cvtColor(gt_m, cv2.COLOR_BGR2GRAY)
        print(f'frame {int(fr)}: gt mask unique values: {np.unique(gt_m)}')
        gt_m[gt_m > 0] = 1
        return gt_m

def parse_args():
    """Parse input arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--data_set', dest='data_set', required=True,
        help='dataset to use: AppleA or Peach or Pear or AppleB', default='AppeleA')
    
    parser.add_argument(
        '--CV',
        help='Cross Validation in SSL',
        default=1, type=int)
    
    parser.add_argument(
        '--ssl_iter',
        help='SSL iteration index to make sure the pretrained model is loaded properly',
        default=0, type=int)
    
    parser.add_argument(
        '--number_gpus',
        help='#GPUS in SSL',
        default=3, type=int)

    parser.add_argument(
        '--gpu_id',
        help='#GPUS in SSL',
        default=0, type=int)
    
    parser.add_argument(
        '--isLocal',
        help='Local or remote Server',
        default=1, type=int)
    
    parser.add_argument(
        '--isTrain',
        help='Local or remote Server',
        default=0, type=int)
    
    parser.add_argument(
        '--pretrained',
        help='Local or remote Server',
        default=0, type=int)
    parser.add_argument(
        '--model_type',
        help='Model type: SL, SSL, SSL-RGR',
        default='SSL', type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    #This script should use only for inference
    #For test-time-augmentation we will use separate script
    #instance segmentation: 0: flower
    #semantic segmentation: 0:flower, 1:background
    instance_model = 0
    apply_regression = 0
    apply_test_aug = 0
    rotation_sample_factor = 4 # #rotation_sample_factor angles from 5 ranges
    vis=1
    apply_rgr = 0
    plot_PR = 1
    if args.isTrain:
        iteration =args.ssl_iter-1
    else:
        iteration =args.ssl_iter
    CV = args.CV
    isLocal = args.isLocal
    model_type = args.model_type
    seg_score_thr = 0.5 #used for vis

    init_params = {}
    init_params['pred_score'] = 0.5 #box_pred_score
    init_params['nms_thr'] = 0.8 #box nms
    init_params['gpu_id'] = args.gpu_id
    init_params['iter_i'] = iteration
    init_params['pretrained'] = args.pretrained
    init_params['default_predictor'] = 0
    init_params['test_aug'] = apply_test_aug
    init_params['angle_ranges'] = [[1, 24], [78, 96], [168, 192], [258, 276], [342, 354]]
    init_params['combine_panoptic'] = 0
    #for semantic soft prediction
    init_params['num_class'] = 2
    init_params['class_ids'] = {'flower':0, 'background':1}
    init_params['rotation_factor'] = rotation_sample_factor
    init_params['data_already_registered'] = False

    #experiment on all dataset
    for i, data_type in enumerate([args.data_set]):
        print(f'starting inference for {data_type}')
        init_params = get_data_dirs(data_type, coderoot, init_params)

        model_path = f'{coderoot}models/{model_type}'
        data_root = f'{coderoot}dataset'
        
        if model_type == 'SSL_RGR':
            init_params['model_path'] = f'{model_path}/{data_type}/CV{CV}/iter{iteration}/model_0019999.pth'
        elif model_type == 'SSL':
            init_params['model_path'] = f'{model_path}/{data_type}/CV{CV}/iter{iteration}/model_0019999.pth'
        elif model_type=='SL':
            init_params['model_path'] = f'{model_path}/AppleA_train/model_0019999.pth'
        else:
            print(f"see the models folder if the checkpoint: {init_params['model_path']} is available or not")

        # init model and load weights
        flower_detector = panoptic_fpn_flower(init_params)
        flower_detector.init_detector()
        softmax = nn.Softmax(dim=0)
        
        #read test set
        test_frames = pd.read_csv(f'{data_root}/ssl_data/{data_type}/CV{CV}/test_0.3.csv', 
                                  header=None, index_col=0)
        pred_ms = []
        gt_ms = []
        avg_time = []
        print(test_frames.values)
        for img_path in test_frames.values[1:]:
            img_path = f"{data_root}/ssl_data/{data_type}/CV{CV}/test_imgs/{os.path.basename(img_path[0])}"
            print(img_path)
            st_time = time.time()
            img=cv2.imread(img_path)
            assert img is not None
            #network: image_height=749, image_width=1333
            # define sliding window size and stride
            if init_params['data'] in ['AppleA','AppleA_train']:
                fr = float(os.path.basename(img_path).split('.')[0].split('IMG_')[-1])
                if fr in [361]:
                    img  = imutils.rotate_bound(img, 90)
                step_size = [img.shape[1] // 16, img.shape[0] // 12]  # [step_w, step_h]: 8,6 or 16,12 # 16 16, 8,8
                window_size = [img.shape[1] // 8, img.shape[0] // 6]  # [w_W, w_H] #4,3 or 8,6

            elif init_params['data'] in ['AppleB', 'Peach']:
                fr = float(os.path.basename(img_path).split('.')[0])
                step_size = [img.shape[1] // 8, img.shape[0] // 8]  # [step_w, step_h]
                window_size = [img.shape[1] // 4, img.shape[0] // 4]  # [w_W, w_H]

            elif init_params['data'] in ['Pear']:
                fr = float(os.path.basename(img_path).split('.')[0].split('_')[-1])
                step_size = [img.shape[1] // 8, img.shape[0] // 8]  # [step_w, step_h]
                window_size = [img.shape[1] // 4, img.shape[0] // 4]  # [w_W, w_H]

            #prepare sliding windows
            SWs = sliding_windows(img=img.copy(), step_size=step_size, window_size=window_size,
                                  gpu_id=init_params['gpu_id'], init_params=init_params, detector=flower_detector)
            #SWs.print_w_stats()
            window_id = 1

            #output mask having actual resolution with padding
            #soft mask should have class specific logits
            soft_mask_pan = np.zeros(img.shape[:2], dtype='float')

            pix_count_mask = copy.deepcopy(soft_mask_pan)
            #make 2 class empty soft mask
            soft_mask_pan = soft_mask_pan.reshape(1, soft_mask_pan.shape[0], soft_mask_pan.shape[1])

            soft_mask = np.vstack([soft_mask_pan, soft_mask_pan])
            soft_mask_ins = np.vstack([soft_mask_pan.copy(), soft_mask_pan.copy()])

            #iterate over window images
            for (x1, y1, x2, y2, window) in SWs.sliding_window():
                #print(f'Window ID: {window_id}')
                #get soft predictions from SL model: single image
                if apply_test_aug:
                    #softmax applied in test_aug
                    pred_mask, pred_results = SWs.test_aug(window)
                else:
                    pred_results = SWs.flower_detector.predict_single(window)
                    pred_mask = pred_results['sem_seg']
                    pred_mask = softmax(pred_mask)
                if args.pretrained:
                    pred_mask = torch.cat((pred_mask[0].unsqueeze(0), pred_mask[9].unsqueeze(0)))
                    print('use pretrained')
 
                assert pred_mask.shape==soft_mask[:, y1:y2,x1:x2].shape
                #print(pred_mask[0])
                normalize_factor = 1
                if apply_regression:
                    #no background, Peach:0.25
                    normalize_factor = 2
                    if instance_model:
                        pred_mask_ins = SWs.instances_to_semantic(window, pred_results['instances'], box_score_thr=init_params['pred_score'])
                        # pred_mask_ins = SWs.apply_regression(pred_mask_ins.clone(), window, seg_score_thr=0.7,
                        # box_score_thr=0.02, area_thr=50, jitter_size=30, jitter_factor=0.1, jitter_boxes=True)

                    else:
                        pred_mask_ins = SWs.apply_regression(pred_mask.clone(), window, seg_score_thr=0.5,
                                                             box_score_thr=init_params['pred_score'], area_thr=100, jitter_size=30, jitter_factor=0.1, jitter_boxes=True)
                else:
                    pred_mask_ins = None


                soft_mask[:,y1:y2, x1:x2] += pred_mask.cpu().numpy()
                if  pred_mask_ins:
                    soft_mask_ins[:,y1:y2, x1:x2] += pred_mask_ins.cpu().numpy()
                #count predicted pixels: to compute average for overlapped pixels
                pix_count_mask[y1:y2,x1:x2] += 1
                assert window.shape[:2]==pred_mask.cpu().numpy().shape[1::], f'found {window.shape[:2]}=={pred_mask.cpu().numpy().shape[1::]}'
                window_id+=1
            print('max values: {}, {}'.format(soft_mask.max(), pix_count_mask.max()))
            final_pred = soft_mask/pix_count_mask
            if pred_mask_ins:
                final_pred_ins = soft_mask_ins/pix_count_mask

            #applying RGR
            print(f'Frame: {fr}, total_windows: {window_id-1}, time elapsed SSL: {time.time()-st_time} secs')
            if apply_rgr:
                final_pred = SWs.rgr_refine(final_pred, img)
            else:
                final_pred = final_pred[0]
                if pred_mask_ins:
                    final_pred_ins = final_pred_ins[0]

                if not plot_PR:
                    final_pred[final_pred>=seg_score_thr] = 1
                    final_pred[final_pred<seg_score_thr] = 0
                    if pred_mask_ins:
                        final_pred_ins[final_pred_ins>=0.26] = 1
                        final_pred_ins[final_pred_ins<0.26] = 0


            assert final_pred.shape[0]==img.shape[0]

            #collect gt and pred to evaluate
            avg_time.append(time.time()-st_time)
            print(f'Frame: {fr}, average time elapsed SSL+RGR: {np.mean(avg_time)} secs')
            print(f'pred mask unique values: {np.unique(final_pred)}')

            pred_ms.append(final_pred)

            gt_m = SWs.get_gt_mask(fr, data_type, dataroot=data_root)
            if fr in [361] and init_params['data']=='AppleA':
                gt_m  = imutils.rotate_bound(gt_m, 90)
                gt_m[gt_m>0] = 1
                assert final_pred.shape==gt_m.shape
            gt_ms.append(gt_m)

        
            if vis:
                SWs.overlay_masks(img_bin=final_pred.copy()>seg_score_thr, im=img, 
                                out_dir=init_params['overlay_out_dir'], 
                                imgname=img_path,dpi=200, box_alpha=0.2,  
                                gt=gt_m.copy(), show_class=True, show_gt_contours=False)
        #evaluation
        #plot PR curve
        if not apply_rgr and plot_PR:
            recall = []
            precision = []
            F1_score = []
            IOU_score = []
            tau_seg = []
            for seg_score_thr in np.arange(0.1,0.8, 0.02):
                eval_mask = evaluate_mask(gpu_id=init_params['gpu_id'])
                iou, F1, Rcll, Prcn , nums = eval_mask.overall_metrics(pred_ms, gt_ms, score_thr=seg_score_thr, area_thr=100)
                print(f'IoU:{iou}, F1:{F1}, Rcll:{Rcll}, Prcn:{Prcn}')
                d = [[f'iter{iteration}', data_type, iou, F1, Rcll, Prcn]]
                print(tabulate(d, headers=["Model", 'Data', "IoU", "F1", "Recall", "Precision"]))
                print(f'Datasets: {data_type}, Score Threshold: {seg_score_thr}, CV: {args.CV}')
                print(f'########################################################')
                
                recall.append(Rcll.cpu().numpy())
                precision.append(Prcn.cpu().numpy())
                F1_score.append(F1.cpu().numpy())
                IOU_score.append(iou.cpu().numpy())
                tau_seg.append(seg_score_thr)
                if max(F1_score)-F1_score[-1] > 0.1:
                    break
                # count stats
                # fl_counter = flower_counter(score_thr=seg_score_thr, area_thr=200, iou_thr=0.2)
                # fl_stats = fl_counter.get_dataset_flower_stats(gt_ms, pred_ms)
                # print(fl_stats)
            score_max_f1 = tau_seg[np.argmax(F1_score)]
            plt.plot(recall, precision)
            print(init_params['model_path'])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # plt.savefig(f'PR_{data_type}_iter{iteration}_{model_type}.png', dpi=200, bbox_inches='tight')
            # #save csv file to generate single figure for all datasets
            
            PR_Data = pd.DataFrame({'precision': precision, 'recall': recall, 'F1':F1_score, 'IOU':IOU_score})
            PR_Data.to_csv(f'eval_results/PR_{data_type}_iter{iteration}_{model_type}_CV{CV}.csv')
            #save tau_seg for training set
            
            tau_seg = pd.DataFrame({'tau_seg': [score_max_f1]})
            # tau_seg.to_csv(f"{data_root}/SSL_Data/{data_type}/CV{CV}/tau_seg_iter{args.ssl_iter-1}.csv", index=False)
            

        else:
            eval_mask = evaluate_mask(gpu_id=init_params['gpu_id'])
            iou, F1, Rcll, Prcn , nums = eval_mask.overall_metrics(pred_ms, gt_ms, area_thr=100)
            print(f'IoU:{iou}, F1:{F1}, Rcll:{Rcll}, Prcn:{Prcn}')
            d = [[f'iter{iteration}', data_type, iou, F1, Rcll, Prcn]]
            print(tabulate(d, headers=["Model", 'Data', "IoU", "F1", "Recall", "Precision"]))
            print(f'Datasets: {data_type}, Score Threshold: {seg_score_thr}, CV: {args.CV}')
            print(f'########################################################')

            # count stats
            # fl_counter = flower_counter(score_thr=seg_score_thr, area_thr=400, iou_thr=0.3)
            # fl_stats = fl_counter.get_dataset_flower_stats(gt_ms, pred_ms)
            # print(fl_stats)