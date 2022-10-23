import copy
import glob
import os.path
import sys
import cv2
import time
import pdb

coderoot = os.path.dirname(os.path.realpath(__file__)).split('ssl_flower_semantic')[0] + 'ssl_flower_semantic'
print(f'coderoot:{coderoot}')
sys.path.insert(0, f"{coderoot}")

import matplotlib.pyplot as plt
import numpy as np
from src.panoptic_flower_model import panoptic_fpn_flower
from utils.utils import get_data_dirs
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
import math

class sliding_windows(object):
    def __init__(self, img, step_size=None, window_size=None):
        self.img=img
        self.step_size=step_size # step_size = (w_step, h_step)
        self.window_size=window_size #window_size = (w, h)

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
    
    def get_gt_mask(self, fr, data_type, dataroot=None):
        assert os.path.exists(dataroot)
    
        gt_path = f'{dataroot}/raw_data/labels/{data_type}/gt_frames'
        if data_type=='Pear':
            gt_m = cv2.imread(gt_path + '/{:03d}.png'.format(int(fr)))
        else:
            gt_m = cv2.imread(gt_path + '/{:03d}.png'.format(int(fr)))

        gt_m = cv2.cvtColor(gt_m, cv2.COLOR_BGR2GRAY)
        print(f'frame {int(fr)}: gt mask unique values: {np.unique(gt_m)}')
        gt_m[gt_m > 0] = 1
        return gt_m

if __name__ == '__main__':
        # This script should use only for inference
    # For test-time-augmentation we will use separate script
    #instance segmentation: 0: flower
    #semantic segmentation: 0:flower, 1:background

    iteration = 1
    score_thr = 0.45 #PR curve: AppleA:0.4(SL), AppleB: 0.6(SSL)
    #mnt_drive = '/media/abubakarsiddique' #'/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62' #'/media/abubakarsiddique'
    #nas_dir =  '/media/NAS/LabFiles/Walden' #'/media/siddique/RemoteServer/LabFiles/Walden'  #'/media/NAS/LabFiles/Walden'

    mnt_drive = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62' #'/media/abubakarsiddique'
    nas_dir =  '/media/siddique/RemoteServer/LabFiles/Walden'  #'/media/NAS/LabFiles/Walden'
    #pdb.set_trace()
    init_params = {}

    init_params['pred_score'] = 0.1
    init_params['nms_thr'] = 0.8
    init_params['gpu_id'] = 1
    init_params['iter_i'] = iteration
    init_params['pretrained'] = 0
    init_params['default_predictor'] = 1
    init_params['test_aug'] = 0
    init_params['angle_ranges'] = [[1, 24], [78, 96], [168, 192], [258, 276], [342, 354]]
    init_params['combine_panoptic'] = 0
    #for semantic soft prediction
    init_params['num_class'] = 2
    init_params['class_ids'] = {'flower':0, 'background':1}


    
    #experiment on all dataset
    for i, data_type in enumerate(['Pear']):#['AppleB', 'AppleA', 'Peach', 'Pear']
        print(f'starting inference for {data_type}')
        init_params = get_data_dirs(data_type, coderoot, init_params)
        #initialize detector
        if init_params['data'] == 'AppleA':
            init_params['model_path'] = f'{mnt_drive}/Panoptic_Models/flower/modified_loss_semi/100_percent/AppleA/iter{iteration}/model_0019999.pth'
        else:    
            init_params['model_path'] = f'{mnt_drive}/Panoptic_Models/flower/modified_loss_semi/100_percent/{data_type}/iter{iteration}/model_0019999.pth'
        print(init_params['model_path'])
        if i==0:
            flower_detector = panoptic_fpn_flower(init_params)
            flower_detector.init_detector()
        softmax = nn.Softmax(dim=0)

        folder = glob.glob(init_params['unlabeled_img_dir']+'/*')
        folder.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))

        pred_ms = []
        gt_ms = []
        for img_path in folder:
            print(img_path)
            st_time = time.time()
            img=cv2.imread(img_path)
            assert img is not None
            #network: image_height=749, image_width=1333
            if init_params['data'] in ['AppleA','AppleA_train']:
                fr = float(os.path.basename(img_path).split('.')[0].split('IMG_')[-1])
                if fr in [361]:
                    img  = imutils.rotate_bound(img, 90)
                step_size = [img.shape[1] // 16, img.shape[0] // 12]  # [step_w, step_h]: 8,6 or 16,12
                window_size = [img.shape[1] // 8, img.shape[0] // 6]  # [w_W, w_H] #4,3 or 8,6

            elif init_params['data'] in ['AppleB', 'Peach']:
                fr = float(os.path.basename(img_path).split('.')[0])
                step_size = [img.shape[1] // 8, img.shape[0] // 8]  # [step_w, step_h]
                window_size = [img.shape[1] // 4, img.shape[0] // 4]  # [w_W, w_H]

            elif init_params['data'] in ['Pear']:
                fr = float(os.path.basename(img_path).split('.')[0].split('_')[-1])
                step_size = [img.shape[1] // 8, img.shape[0] // 8]  # [step_w, step_h]
                window_size = [img.shape[1] // 4, img.shape[0] // 4]  # [w_W, w_H]

            # if vis_window:
            #     cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #     cv2.resizeWindow("image", img.shape[1] // 4, img.shape[0] // 3)
            #if fr in [356]:
            #prepare sliding windows
            SWs = sliding_windows(img=img.copy(), step_size=step_size, window_size=window_size)
            SWs.print_w_stats()
            window_id = 1

            #output mask having actual resolution with padding
            #soft mask should have class specific logits
            #soft_mask = SWs.get_padded_mask()
            soft_mask = np.zeros(img.shape[:2], dtype='float')
            pix_count_mask = copy.deepcopy(soft_mask)
            #make 2 class empty soft mask
            soft_mask = soft_mask.reshape(1, soft_mask.shape[0], soft_mask.shape[1])
            soft_mask = np.vstack([soft_mask, soft_mask])
            #pdb.set_trace()
            #iterate over window images
            for (x1, y1, x2, y2, window) in SWs.sliding_window():
                # if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                #     pdb.set_trace()
                #print('window size: {}'.format(window.shape))
                #print('window id: {}'.format(window_id))
                #print(f'RGB rectangle x1,y1,x2,y2: {[x1, y1, x2, y2]}')
                #get soft predictions from SL model: single image
                pred_results = flower_detector.predict_single(window)
                pred_mask = pred_results['sem_seg']
                
                
                if pred_mask is not None:
                    assert pred_mask.shape==soft_mask[:, y1:y2,x1:x2].shape
                    #print(pred_mask[0])
                    pred_mask = softmax(pred_mask)
                    soft_mask[:,y1:y2, x1:x2] += pred_mask.cpu().numpy()
                    #count predicted pixels: to compute average for overlapped pixels
                #print(pix_count_mask[y1:y2,x1:x2].shape)
                pix_count_mask[y1:y2,x1:x2] += 1
                assert window.shape[:2]==pred_mask.cpu().numpy().shape[1::]
                window_id+=1
            print('max values: {}, {}'.format(soft_mask.max(), pix_count_mask.max()))
            #pdb.set_trace()
            final_pred = soft_mask/pix_count_mask
            #print('max values: {}, {}'.format(final_pred.max(), pix_count_mask.max()))

            # #TODO: save both classes score map before applying RGR
            # final_pred = torch.tensor(final_pred).cuda()
            # final_pred = softmax(final_pred)
            # final_pred = final_pred.argmax(dim=0).cpu().numpy()
            # final_pred[final_pred==0] = 255
            # final_pred[final_pred==1] = 0

            #final_pred = final_pred.cpu().numpy()
            #final_pred = SWs.remove_pad_scores(final_pred, img.shape)
            #final_pred = SWs.remove_pad(final_pred, img.shape)
            #SWs.save_scores(final_pred, init_params['out_dir'], img_path)
            
    
            #final_pred = SWs.remove_pad(final_pred, img.shape)
            # final_pred = torch.tensor(final_pred).cuda()
            # final_pred = softmax(final_pred)
            # final_pred = final_pred[0].cpu().numpy() # 0: flower, 1: bck
            final_pred = final_pred[0]
            final_pred[final_pred>=score_thr] = 1 #Peach: 0.4, Pear:0.25, AppleB:0.4, AppleA: 0.45 
            final_pred[final_pred<score_thr] = 0
            # final_pred[final_pred>0] = 1
            
            assert final_pred.shape[0]==img.shape[0]

            #collect gt and pred to evaluate
            print(f'Frame: {fr}, time elapsed SSL+RGR: {time.time()-st_time} secs')
            print(f'pred mask unique values: {np.unique(final_pred)}')
            # final_pred = SWs.apply_area_filter(final_pred, area_thr=200)
            pred_ms.append(final_pred)

            gt_m = SWs.get_gt_mask(fr, data_type, img_path, nas_dir)
            if fr in [361] and init_params['data']=='AppleA':
                gt_m  = imutils.rotate_bound(gt_m, 90)
                gt_m[gt_m>0] = 1
                assert final_pred.shape==gt_m.shape

            # gt_m = SWs.apply_area_filter(gt_m, area_thr=200)
            gt_ms.append(gt_m)

            #save_path = os.path.join(init_params['out_dir'], os.path.basename(img_path))
            #SWs.save_final_pred(save_path, final_pred)
            #TODO: save difference of gt_mask and predicted semantic mask
            eval_mask = evaluate_mask()
            iou, F1, Rcll, Prcn, nums = eval_mask.overall_metrics([final_pred], [gt_m])

            print(f'IoU:{iou}, F1:{F1}, Rcll:{Rcll}, Prcn:{Prcn}')
            d = [[f'iter{iteration}', data_type, "iou: %.3f"%iou,  "F1: %.3f"%F1,  "Rcll: %.3f"%Rcll,  "Prcn: %.3f"%Prcn]]
            print(tabulate(d, headers=["Model", 'Data', "IoU", "F1", "Recall", "Precision"]))

            
            SWs.overlay_masks(img_bin=final_pred.copy(), im=img, out_dir=init_params['overlay_out_dir'], imgname=img_path,
                            dpi=200, box_alpha=0.2,  gt=gt_m.copy(), evaluation=d[0], nums=nums, show_class=True, show_gt_contours=True)

        #evaluation
        #plot PR curve
        # recall = []
        # precision = []
        # for score_thr in np.arange(0.1,0.96, 0.02):
        #     eval_mask = evaluate_mask()
        #     iou, F1, Rcll, Prcn , nums = eval_mask.overall_metrics(pred_ms, gt_ms, score_thr=score_thr, area_thr=200)
        #     print(f'IoU:{iou}, F1:{F1}, Rcll:{Rcll}, Prcn:{Prcn}')
        #     d = [[f'iter{iteration}', data_type, iou, F1, Rcll, Prcn]]
        #     print(tabulate(d, headers=["Model", 'Data', "IoU", "F1", "Recall", "Precision"]))
        #     print(f'Datasets: {data_type}, Score Threshold: {score_thr}')
        #     recall.append(Rcll.cpu().numpy())
        #     precision.append(Prcn.cpu().numpy())
        # plt.plot(recall, precision)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.savefig(f'PR_{data_type}_iter{iteration}.png', dpi=200, bbox_inches='tight')