import cv2
import numpy as np
import torch
from tabulate import tabulate
import pdb
import matplotlib.pyplot as plt
import glob
import os
import pycocotools.mask as mask_util
from skimage.morphology import label

class evaluate_mask(object):
    '''This implementation follow the baseline method: RGR, PASCAL VOC'''
    def __init__(self, pred_m=None, gt_m=None, gpu_id=0):
        self.gpu_id = 0
        if pred_m is not None:
            self.pred_m = torch.tensor(pred_m, dtype=torch.uint8, device=f'cuda:{self.gpu_id}')
            self.gt_m = torch.tensor(gt_m, dtype=torch.uint8, device=f'cuda:{self.gpu_id}')
        else:
            self.pred_m = None
            self.gt_m = None
        self.PN = 0
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def apply_area_filter(self, bin_mask, area_thr=400):
        label_mask = label(bin_mask>0)
        img_props_ind = list(np.unique(label_mask))
        #print(f'segment indexes: {img_props_ind}')
        img_props_ind.remove(0)
        #print(f'total flower found {len(img_props_ind)}')
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

    def convert_to_tensor(self, mask_bin):
        return torch.tensor(mask_bin, dtype=torch.uint8, device=f'cuda:{self.gpu_id}')

    def compute_TP(self):
        return self.pred_m * self.gt_m

    def compute_FP(self):
        return self.pred_m*(~(self.gt_m>0)).type(torch.uint8)

    def compute_FN(self):
        return (~(self.pred_m>0)).type(torch.uint8) * self.gt_m

    def get_img_metrics(self):
        positive_num = (self.gt_m == 1).sum().type(torch.float)
        tp_num = (self.compute_TP() == 1).sum().type(torch.float)
        fp_num = (self.compute_FP() == 1).sum().type(torch.float)
        fn_num = (self.compute_FN() == 1).sum().type(torch.float)

        self.PN += positive_num
        self.TP += tp_num
        self.FP += fp_num
        self.FN += fn_num

    def overall_metrics(self, pred_ms, gt_ms, score_thr=None, area_thr=None):
        assert len(pred_ms)==len(gt_ms)
        for pred_m, gt_m in zip(pred_ms, gt_ms):
            self.pred_m = pred_m.copy()
            self.gt_m = gt_m.copy()
            if score_thr is not None:
               self.pred_m[self.pred_m>=score_thr] = 1
               self.pred_m[self.pred_m!=1] = 0
            if area_thr is not None:
                self.pred_m = self.apply_area_filter(self.pred_m, area_thr=area_thr)
                self.gt_m = self.apply_area_filter(self.gt_m, area_thr=area_thr)

            self.pred_m = self.convert_to_tensor(self.pred_m)
            self.gt_m = self.convert_to_tensor(self.gt_m)
            self.get_img_metrics()

        assert  self.PN==self.TP+self.FN
        print(f'PN: {self.PN}, TP: {self.TP}, FN: {self.FN}, FP: {self.FP}')

        self.Rcll_avg = torch.div(self.TP, self.PN) 
        self.Prcn_avg = torch.div(self.TP, (self.TP + self.FP))
        self.IoU_avg = torch.div(self.TP, (self.TP + self.FP + self.FN))
        # F1 Score = 2*(Recall * Precision) / (Recall + Precision)
        self.F1_avg = torch.div(2 * (self.Rcll_avg * self.Prcn_avg),
                                (self.Rcll_avg + self.Prcn_avg))

        return  self.IoU_avg, self.F1_avg, self.Rcll_avg, self.Prcn_avg, [self.PN, self.TP, self.FN, self.FP]


if __name__ == '__main__':

    gt_path = '/media/siddique/RemoteServer/LabFiles/Walden/trainTestSplit/test/rawData/bMasks'
    pred_folder = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/flower/AppleA'
    folder = glob.glob(pred_folder+'/panet_pred/*')
    folder.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
    pred_ms = []
    gt_ms = []
    for pred_path in folder:
        fr = float(os.path.basename(pred_path).split('.')[0].split('IMG_')[-1])
        #print('Frame: {}'.format(int(fr)))
        pred_m = cv2.imread(pred_path)
        pred_m = cv2.cvtColor(pred_m, cv2.COLOR_BGR2GRAY)
        print('pred mask unique values: {}', np.unique(pred_m))
        pred_m[pred_m <= 7] = 0
        pred_m[pred_m > 7] = 1
        pred_ms.append(pred_m)

        gt_m = cv2.imread(gt_path+'/{:03d}.png'.format(int(fr)))
        gt_m = cv2.cvtColor(gt_m, cv2.COLOR_BGR2GRAY)
        print('gt mask unique values: {}', np.unique(gt_m))
        gt_m[gt_m > 0] = 1
        gt_ms.append(gt_m)
    #eval_mask = evaluate_mask(gt_m=gt_m, pred_m=pred_m)
    #Rcll, Prcn, iou = eval_mask.get_img_metrics()
    eval_mask = evaluate_mask()
    iou, F1, Rcll, Prcn = eval_mask.overall_metrics(pred_ms, gt_ms)

    #print([Rcll, Prcn, iou])
    d = [["SL", "AppleA", iou, F1, Rcll, Prcn]]
    print(tabulate(d, headers=["Model", 'Data', "IoU", "F1", "Recall", "Precision"]))
