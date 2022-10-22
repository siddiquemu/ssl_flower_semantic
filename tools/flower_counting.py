
from skimage.morphology import label
from skimage.measure import regionprops
import pycocotools.mask as mask_util
import numpy as np

class flower_counter(object):
    def __init__(self, score_thr=None, area_thr=200, iou_thr=0.5):
        self.seqtp = 0
        self.seqfp = 0
        self.seqfn = 0
        self.seqgt = 0
        self.score_thr = score_thr
        self.area_thr = area_thr
        self.iou_thr = iou_thr

    @staticmethod
    def semantic_to_instance(sem_mask=None, score_thr=None, area_thr=200):
        """
        Function to return RLE of instance masks
        Args:
            sem_mask: binary semantic mask
        Return:
            instance_mask: list of RLEs of Instance mask
        """
        RLEs = []
        label_mask = label(sem_mask>score_thr)
        img_props_ind = list(np.unique(label_mask))
        img_props_ind.remove(0)
        #print(f'total flower found {len(img_props_ind)}')
        for i in img_props_ind:
            #compute instance area
            ins_mask = np.zeros(label_mask.shape, dtype='uint8')
            ins_mask[label_mask==i] = 1
            rle = mask_util.encode(np.asfortranarray(ins_mask))
            area = mask_util.area(rle)
            if area>area_thr:
                RLEs.append(rle)
        return RLEs

    def mask_iou(self, rle_i, rle_j):
        return mask_util.iou([rle_i], [rle_j], [False])[0][0]

    def get_frame_flower_stats(self):
        self.tmptp = 0
        self.tmpfp = 0
        self.tmpfn = 0
        num_associations = 0
        for row, gg in enumerate(self.gt_rles):
          for col, tt in enumerate(self.pred_rles):
            c = self.mask_iou(gg, tt)
            if c > self.iou_thr:
              # true positives are only valid associations
              self.tmptp += 1
              num_associations += 1
              #break #since one one gt might associate with multiple predicted masks for the clusters of flowers
        # update sequence data
        self.seqtp += self.tmptp
        self.seqfp += len(self.pred_rles) - self.tmptp
        self.seqfn += len(self.gt_rles) - num_associations
        self.seqgt += len(self.gt_rles)

    def get_dataset_flower_stats(self, gts, preds):
        for gt, pred in zip(gts, preds):
            self.gt_rles = self.semantic_to_instance(gt, score_thr=0, area_thr=self.area_thr)
            self.pred_rles = self.semantic_to_instance(pred, score_thr=self.score_thr, area_thr=self.area_thr)
            self.get_frame_flower_stats()
        return {'nGT':self.seqgt, 'nTP':self.seqtp, 'nFP':self.seqfp, 'nFN':self.seqfn}
