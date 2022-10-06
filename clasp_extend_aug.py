from pycocotools.coco import COCO
import numpy as np
#from skimage import measure
import matplotlib.pyplot as plt
#from matplotlib.patches import Polygon
import pylab
import json
import urllib
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
import cv2
import os
from collections import OrderedDict
import collections
from clasp2coco import define_dataset_dictionary, Write_AnnotationInfo, Write_ImagesInfo, Write_To_Json
import torchvision.transforms as T
from PIL import Image

def get_frame_anns(coco_clasp, im_id):
    # return anns for an image
    Cur_imgId = coco_clasp.getImgIds(imgIds=[im_id])
    img = coco_clasp.loadImgs(Cur_imgId)[0]
    annIds = coco_clasp.getAnnIds(imgIds=img['id'])
    #annIds.extend(coco_clasp.getAnnIds(imgIds=img['id'], catIds=2, iscrowd=0))
    allanns = coco_clasp.loadAnns(annIds)
    return allanns

def get_mixed_aug_images(orig_img):
    imgs = []
    jitter = T.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.2, hue=0.05)
    color_jitted_imgs = [jitter(orig_img) for _ in range(1)]
    color_jitted_imgs.append(orig_img)

    for i, imc in enumerate(color_jitted_imgs):
        imgs.append(imc)

        blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        imgs.append(blurrer(imc))
    print(f'size of the other aug: {len(imgs)}')
    return imgs

#read COCO json of any SSL iteration-0 where rotation is the only augmentation in pseudo-labels
storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62'
#storage = '/media/abubakarsiddique'
dataset = 'clasp2'

data_path = f'{storage}/tracking_wo_bnw/data/CLASP/test_augMS_gt_score/iter0'

#rotation aug data dir
coco_json = f'{data_path}/{dataset}_test_aug_0.json'
rot_aug_img_path = f'{data_path}/img1_0'
coco_clasp = COCO(coco_json)

#modified data dir
AugImgDir = f'{data_path}/img1_0_mixed_aug'
dataJSONfile = f'{data_path}/{dataset}_test_aug_0_mixed_aug.json'
if not os.path.exists(AugImgDir):
    os.makedirs(AugImgDir)
    
catIds = [2]
imgIds_bag = coco_clasp.getImgIds(catIds=catIds)
print('total images of bag is', len(imgIds_bag))
#print('Their images ids are', imgIds_bag)

catIds = [1]
imgIds_person = coco_clasp.getImgIds(catIds=catIds)
print('total images of person is', len(imgIds_person))
#print('Their images ids are', imgIds_person)
#iterate over the rotationally augmented frames and create new aug. such as color jitter, motion blur: torch>=1.7
angleSet = [0, 186, 90, 270]
imgIds = coco_clasp.getImgIds()
print(f'total images: {len(imgIds)}')
rot_imgs = []
for fr in imgIds:
    angle = int(str(fr)[-3:])
    if angle in angleSet:
        rot_imgs.append(fr)
print(f'rot images in extended aug. :{len(rot_imgs)}')
#update the existing JSON for new frames and labels
clasp_mixed_aug = define_dataset_dictionary()
# color jitter and motion blur will create: len(rot_imgs)x4 total frames
fr_new = 1
ann_id_new = 1
frame_ids = []
ann_ids = []

for fr_id in rot_imgs:
    Cur_imgId = coco_clasp.getImgIds(imgIds=[fr_id])
    img = coco_clasp.loadImgs(Cur_imgId)[0]
    annIds = coco_clasp.getAnnIds(imgIds=img['id'])
    allAnns = coco_clasp.loadAnns(annIds)
    print(f'total labels: {len(allAnns)}')
    #read img
    labeled_img_dir = '{}/{}'.format(rot_aug_img_path, img['file_name'])
    orig_img = Image.open(labeled_img_dir)
    imgs = get_mixed_aug_images(orig_img)

    theta = int(str(fr_id)[-3:])

    print(f'training examples: frame: {fr_id}, #detection: {len(allAnns)}, angle: {theta}')
    for i, im_aug in enumerate(imgs):
        # save image info
        imgIdnew = int(f'{fr_id}{i}')
        #imgIdnew = 10000 * int('%06d' % fr_) + theta_new
        frame_ids.append(imgIdnew)  # populate frame ids to verify the process terminate
        imgname = '{:08d}.png'.format(imgIdnew)
        img_write_path = AugImgDir + '/' + imgname
        # if fr_num in self.semi_frames:
        #     self.vis_gt(self.rot_imgs_dict[theta], fr_det, gt_vis_path=self.vis_path, imname=img_write_path)

        print(f'Writing image {imgname}')
        im_aug = cv2.cvtColor(np.array(im_aug), cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_write_path, im_aug)
        #im_aug.save(img_write_path, "PNG")
        clasp_mixed_aug = Write_ImagesInfo(im_aug, imgname, int(imgIdnew), clasp_mixed_aug)

        # save anns
        for ib, ann_id in enumerate(annIds):
            ann = coco_clasp.loadAnns(ann_id)[0]
            box = ann['bbox']
            segm = ann['segmentation']
            score = ann['instance_certainty']
            catID = ann['category_id']
            area = ann['area']
            # # since only consider the cluster modes
            # if (box[6] >= self.cluster_score_thr[0] and box[7] == 1) or \
            #         (box[6] >= self.cluster_score_thr[1] and box[7] != 1):
            #
            #     bboxfinal = [round(x, 2) for x in box[2:6]]
            #     mask = fr_mask[ib]
            #     area = mask_util.area(mask)  # polygon area
            #     assert len(box) == 9, 'box {}'.format(box)
            #     # [fr, i, bbox[0], bbox[1], w, h, score, classes[i]]
            #     if self.database in ['clasp1', 'clasp2']:
            #         if box[7] == 1:
            #             catID = 1
            #         else:
            #             catID = 2
            #     else:
            #         catID = 1
            #     # score: cluster_score or cluster_mode regressed score
            #     score = box[6]
                # 1000 * int('%06d' % (fr_num+ib+1)) + theta
            annID = int(f'{ib+1}{imgIdnew}')  # fr_num is unique in multiple precesses
            ann_ids.append(annID)

            # segmPolys = []  # mask['counts'].decode("utf-8") #[]
            # bmask = mask_util.decode(mask)
            # contours = measure.find_contours(bmask, 0.5)
            # # contours = cv2.findContours(bmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # for contour in contours:
            #     contour = np.flip(contour, axis=1)
            #     segmentation = contour.ravel().tolist()
            #     # print(f'segm poly size {len(segmentation)}')
            #     if len(segmentation) > 0:
            #         segmPolys.append(segmentation)
            # assert int(imgIdnew) == int(os.path.basename(img_write_path).split('.')[0])
            # # save annotation for for each image
            # if len(segmPolys) > 0:
            clasp_mixed_aug = Write_AnnotationInfo(box, segm, int(imgIdnew),
                                                          int(annID), catID, int(area), clasp_mixed_aug,
                                                          instance_certainty=score)
    fr_new += 1

#save the final JSON and start training
assert len(frame_ids) == len(set(frame_ids))
assert len(ann_ids) == len(set(ann_ids))

Write_To_Json(dataJSONfile, clasp_mixed_aug)
