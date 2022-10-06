from json.tool import main
import sys
sys.path.insert(0, '/home/siddique/PANet/rgr-public/Python')
from sliding_windows import sliding_windows
import cv2
import numpy as np
import matplotlib.pyplot as plt

from clasp2coco import define_flower_dataset_dictionary, Write_To_Json, Proxy2RealObj, \
    Write_ImagesInfo, Write_AnnotationInfo, load_clasp_json, get_frame_anns
import pycocotools.mask as mask_util
from skimage import measure
import os
import glob
import random
from PIL import ImageColor
from skimage.morphology import label
from skimage.measure import regionprops

from PIL import Image
import torch
import torchvision.transforms as T
import imutils
import copy
import pdb

class Proposal_Box(object):
    def __init__(self, out_dir, color_jitter=False, rotation_aug=False, img_list=None, isLabeled=False, vis=False):
        self.imglist = img_list
        self.img_mask_labels = None
        self.img_bin = None
        self.img=None
        self.isLabeled = isLabeled
        self.dataset_flower = define_flower_dataset_dictionary()
        self.vis = vis
        self.out_dir = out_dir
        self.ann_id_count = 1
        if self.isLabeled:
            self.img_id_count = 1
        else:
            self.img_id_count = 46509
        self.verbose=False
        self.apply_color_jitter = color_jitter
        self.jitted_imgs = None
        self.rotation_aug = rotation_aug
        if self.isLabeled:
            if self.apply_color_jitter or self.rotation_aug:
                self.out_img_path = f'{self.out_dir}/trainFlowerAug'
            else:
                self.out_img_path = f'{self.out_dir}/trainFlower'
        else:
            self.out_img_path = f'{self.out_dir}/unlabeledFlowerPear'

        if not os.path.exists(self.out_img_path):
            os.makedirs(self.out_img_path)
            
        self.instance_label_path = f'{self.out_dir}/instance_labels'
        if not os.path.exists(self.instance_label_path):
            os.makedirs(self.instance_label_path)
        self.rgb_mask  = None
        self.ranges = [[6, 24], [78, 96], [168,192], [258, 276], [342, 354]]

    @staticmethod
    def get_random_angles(ranges=None, factor=None):
        angleSet = [0]
        # angle = 180 will be applied during training
        for intvl in ranges:
            angleSet += random.sample(range(intvl[0], intvl[1]), factor)
        return angleSet

    #%%function for converting contours format to segmentation format
    @staticmethod
    def cnt2seg(contours):
        segmentation = [x.flatten() for x in contours]
        segmentation= [[float(y) for y in x] for x in segmentation if len(x) >=6]
        return segmentation

    @staticmethod
    def pil2opencv(img):
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def show_img(img_bin, img_mask_labels):
        plt.imshow(img_bin)
        plt.show()

        plt.imshow(img_mask_labels)
        plt.show()

    def get_random_color(self, num_of_colors):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(num_of_colors)]
        return color

    def save_rgb_mask(self):
        """Useful to learn a panoptic model
        """
        img_mask_rgb = np.zeros(self.img.shape, dtype='uint8')
        color = self.get_random_color(len(np.unique(self.img_mask_labels)))
        for id in np.unique(self.img_mask_labels):
            R, G, B = ImageColor.getcolor(color[id], "RGB")
            img_mask_rgb[self.img_mask_labels==id,0] = R
            img_mask_rgb[self.img_mask_labels==id,1] = G
            img_mask_rgb[self.img_mask_labels==id,2] = B

        cv2.imwrite(f'{self.instance_label_path}/{os.path.basename(self.img_write_path)}', img_mask_rgb)
    

    def get_label_masks(self, img_bin):
        #useful for panoptic segmentation mask and bounding boxes from a binary mask'
        self.img_mask_labels = label(img_bin)
        if self.vis:
            self.show_img(self.img_bin, self.img_mask_labels)
        return self.img_mask_labels
    
    def get_label_boxes(self, label_mask):
        self.fr_boxs = []
        self.fr_masks = []
        self.areas = []
        h,w=label_mask.shape
        #pdb.set_trace()
        #cv2 version 3.x
        _, contours_all_buf, hier_buf = cv2.findContours(label_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)   ##internal and external contours
        if len(contours_all_buf)>0:
            contours_all=[]
            hier=[]
            for i in range(len(contours_all_buf)):
                if len(contours_all_buf[i])>=6:
                    contours_all.append(contours_all_buf[i])
                    hier.append(hier_buf[0,i,:])
                    
            hier=[hier]
            segmentation_all = self.cnt2seg(contours_all)   
            contours_inner=[] 
            for i in range(len(segmentation_all)):
                exists_parent=(hier[0][i][3]!=-1)
                empty_mask = np.zeros((h,w),dtype='uint8')
                if not exists_parent and  cv2.contourArea(contours_all[i])>100:
                    self.fr_masks.append(segmentation_all[i])
                    self.areas.append(cv2.contourArea(contours_all[i]))
                    self.fr_boxs.append(cv2.boundingRect(contours_all[i]))                                                    #
                else:
                    contours_inner.append(contours_all[i])

    def get_label_boxes_conn_comp(self, label_mask):
        self.fr_boxs = []
        self.fr_masks = []
        self.areas = []
        label_mask = label(label_mask)
        img_props_ind = list(np.unique(label_mask))
        img_props_ind.remove(0)
        print(f'total flower found {len(img_props_ind)}')
        for i in img_props_ind:
            empty_mask = np.zeros(label_mask.shape, dtype='uint8')
            empty_mask[label_mask==i] = 255
            rle = mask_util.encode(np.asfortranarray(empty_mask))

            if mask_util.area(rle)>200:
                self.fr_masks.append(rle)
                #print(f'{i}: Bounding box = {mask_util.toBbox(rle)}')
                self.fr_boxs.append(mask_util.toBbox(rle))
                self.areas.append(mask_util.area(rle))

    def get_annotation(self, mask, image=None):
        self.fr_boxs = []
        self.fr_masks = []
        _, contours, _= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            # Valid polygons have >= 6 coordinates (3 points)
            if contour.size >= 6:
                segmentation.append(contour.flatten().tolist())
        RLEs = mask_util.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
        RLE = mask_util.merge(RLEs)
        # RLE = mask_util.encode(np.asfortranarray(mask))
        area = mask_util.area(RLE)
        [x, y, w, h] = cv2.boundingRect(mask)

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.drawContours(image, contours, -1, (0,255,0), 1)
            cv2.rectangle(image,(x,y),(x+w,y+h), (255,0,0), 2)
            cv2.imshow("", image)
            cv2.waitKey(1)

        return segmentation, [x, y, w, h], area 

    def get_rgb_img(self, rgb_w):
        self.jitted_imgs = []
        if self.apply_color_jitter:
            # jitted_imgs are pil image
            self.img = Image.fromarray(rgb_w)
            print(f'apply color jitter to augment the training data')
            jitter = T.ColorJitter(brightness=0, contrast=0.7, saturation=0.4, hue=0.2)
            self.jitted_imgs = [jitter(self.img) for _ in range(3)]
            self.jitted_imgs.append(self.img)
        else:
            self.jitted_imgs.append(rgb_w)
            #print(f'use only original image')
            
    def get_rotated_labels(self, img, angle=None):
        """fr_masks are in rle format
        """
        label_mask  = copy.deepcopy(self.img_bin)
        label_mask_rot = imutils.rotate_bound(label_mask, angle)
        label_mask_rot[label_mask_rot>0] = 255
        label_mask_rot = label(label_mask_rot)
        if self.vis:
            self.show_img(label_mask, label_mask_rot)

        assert img.shape[:2]==label_mask_rot.shape, f'{img.shape[:2]}=={label_mask_rot.shape}'
        # update fr_boxs and fr_masks
        self.get_label_boxes_conn_comp(label_mask_rot)

    def update_anns(self):
        #TODO: in case of scale augmentation do not use same anns for all augmented frames
        if self.rotation_aug:
            self.angle_set = self.get_random_angles(ranges=self.ranges, factor=2)
            print(f'apply rotation aug: {self.angle_set}')
        else:
            self.angle_set = [0]

        for angle in self.angle_set:
            if angle>0:
                print(f'rotated by {angle}')
                #rotate image: self.img will be rotated inplace
                img_rot = imutils.rotate_bound(self.img.copy(), angle)
                #rotate anns: mask is in rle format
                # self.fr_boxs, self.fr_masks will be updated inplace for each orientation 
                self.get_rotated_labels(img_rot, angle)
            else:
                img_rot = self.img.copy()

            #save image info
            imgIdnew = 10000 * int('%06d' % self.img_id_count)
            imgname = '{:08d}.png'.format(imgIdnew)
            self.img_write_path = f'{self.out_img_path}/{imgname}'

            #if self.verbose:
            print(f'Writing image {imgname}, #anns: {len(self.fr_boxs)}')
            cv2.imwrite(self.img_write_path, img_rot)
            self.dataset_flower = Write_ImagesInfo(img_rot, imgname, int(imgIdnew), self.dataset_flower)
            self.img_id_count += 1

            for ib, box in enumerate(self.fr_boxs):
                bboxfinal = [round(x, 2) for x in box[0:4]]
                #segmPolys = self.fr_masks[ib]
                mask = self.fr_masks[ib]
                #area = mask_util.area(mask)#rle area
                area = self.areas[ib]
                catID = 1
                score = 1
                annID = 1000 * int('%06d' % (self.ann_id_count))
                self.ann_id_count += 1

                segmPolys = []  # mask['counts'].decode("utf-8") #[]
                if self.isLabeled:
                    bmask = mask_util.decode(mask)
                    contours = measure.find_contours(bmask, 0.5)
                    for contour in contours:
                        contour = np.flip(contour, axis=1)
                        segmentation = contour.ravel().tolist()
                        if len(segmentation)>0:
                            segmPolys.append(segmentation)
                    assert int(imgIdnew)== int(os.path.basename(self.img_write_path).split('.')[0])
                    #save annotation for for each image
                    if len(segmPolys)>0:
                        self.dataset_flower = Write_AnnotationInfo(bboxfinal, segmPolys, int(imgIdnew),
                                                            int(annID), catID, int(area), self.dataset_flower,
                                                                    instance_certainty=score)
                else:
                    self.dataset_flower = Write_AnnotationInfo(bboxfinal, segmPolys, int(imgIdnew),
                                    int(annID), catID, int(area), self.dataset_flower,
                                            instance_certainty=score)
    def get_coco_structure(self):
        # iterate over color augmented images
        # anns scale remain unchanged
        for _, jitt_pil_im in enumerate(self.jitted_imgs):
            if self.apply_color_jitter:
                self.img = self.pil2opencv(jitt_pil_im)
            else:
                self.img = jitt_pil_im
            self.update_anns()

    def get_all_anotations(self, rgb_list, gt_list, fr_num):
        """self.imglist is the list of high resolution images
        """
        for rgb_w, gt_w in zip(rgb_list, gt_list):
            #label_mask = self.get_label_masks(gt_w)
            # trturn w_box, w_mask
            #self.get_label_boxes(gt_w)
            self.get_label_boxes_conn_comp(gt_w.copy())

            if len(self.fr_boxs)>0:
                self.get_rgb_img(rgb_w)
                self.img_bin = gt_w.copy()
                self.get_coco_structure()
                #to visualize
                #self.save_rgb_mask()
            print(f'Frame: {fr_num}, total augmented frames: {self.img_id_count}')

    def save_anns(self):
        #save flower labels for AppleA train set only
        if self.isLabeled:
            Write_To_Json(f'{self.out_dir}/instances_train_2021.json', self.dataset_flower)
        else:
            Write_To_Json(f'{self.out_dir}/instances_unlabeled_2021_Pear.json', self.dataset_flower)

def get_all_data_dirs(all_args):
    """Dirs of all high resolution frames
    """
    all_args['all_data'] = {}

    all_args['all_data']['AppleA_train'] = all_args['NAS'] + '/trainTestSplit/train/dataFormattedProperly'
    #all_args['all_data']['AppleA_train'] = '/media/siddique/CLASP2019/flower/AppleA_train_refined_gt/PixelLabelData_1'
    all_args['all_data']['AppleA'] = all_args['NAS'] + '/trainTestSplit/test/dataFormattedProperly'
    all_args['all_data']['AppleB'] = all_args['NAS'] + '/otherFlowerDatasets/AppleB/dataFormattedProperly'
    all_args['all_data']['Pear'] = all_args['NAS'] + '/otherFlowerDatasets/Pear/dataFormattedProperly'
    all_args['all_data']['Peach'] = all_args['NAS'] + '/otherFlowerDatasets/Peach/dataFormattedProperly'
    for d_name, path in all_args['all_data'].items():
        assert os.path.exists(path), 'not exist {}: {}'.format(d_name, path)
    return all_args

if __name__=='__main__':
    """This script will read all the labeled and unlabeled high resolution frames
    and run the strided sliding window to get JSON training files
    - Output size: n_theta*n_cropped_frames*n_input_frames
    """
    apply_color_jitter = False
    apply_rotation_aug = False
    apply_sliding_window = True
    isLabeled = False
    
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61'
    cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'
    # storage = '/media/abubakarsiddique'
    # nas_dir = '/media/NAS/Walden'
    # cfg_file = '/media/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'
    all_args={}
    all_args['NAS'] = '/media/siddique/RemoteServer/LabFiles/Walden'
    all_args['out_dir'] = f'{storage}/tracking_wo_bnw/data/flower/train_gt_sw'

    if not os.path.exists(all_args['out_dir']):
        os.makedirs(all_args['out_dir'])
    all_args['out_dir_vis'] = f'{storage}/tracking_wo_bnw/data/flower/train_gt_sw/vis_anns'
    all_args = get_all_data_dirs(all_args)

    labeler = Proposal_Box(out_dir=all_args['out_dir'], rotation_aug=apply_rotation_aug,
                            color_jitter=apply_color_jitter, isLabeled=isLabeled, vis=False)

    for data_type in ['Pear']:#, 'AppleB', 'Peach', 'Pear']: #['AppleA_train']:#,
        data_dir = all_args['all_data'][data_type]
        folder = glob.glob(data_dir+'/flowers/*')
        folder.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        w_id=1
        for img_path in folder:
            rgb_frames = []
            gt_frames = []
            img=cv2.imread(img_path)

            if data_type in ['AppleA', 'AppleA_train']:
                fr = float(os.path.basename(img_path).split('.')[0].split('IMG_')[-1])
                step_size = [img.shape[1] // 8, img.shape[0] // 6]  # [step_w, step_h]
                window_size = [img.shape[1] // 4, img.shape[0] // 3]  # [w_W, w_H]

            elif data_type in ['AppleB', 'Peach']:
                fr = float(os.path.basename(img_path).split('.')[0])
                step_size = [img.shape[1] // 8, img.shape[0] // 8]  # [step_w, step_h]
                window_size = [img.shape[1] // 4, img.shape[0] // 4]  # [w_W, w_H]

            elif data_type in ['Pear']:
                fr = float(os.path.basename(img_path).split('.')[0].split('_')[-1])
                step_size = [img.shape[1] // 8, img.shape[0] // 8]  # [step_w, step_h]
                window_size = [img.shape[1] // 4, img.shape[0] // 4]  # [w_W, w_H]
            else:
                print(f'dataset {data_type} is not exist')
            
            SWs = sliding_windows(img=img.copy(), step_size=step_size, window_size=window_size)
            gt_m = SWs.get_gt_mask(fr, data_type, img_path, all_args['NAS'])
            gt_m = SWs.pad_img(gt_m, step_size)

            print(f'input shape: {SWs.padded_shape}, gt_m shape: {gt_m.shape}')
            for (x, y, window) in SWs.sliding_window_pad():
                if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                    pdb.set_trace()
                #print('window size: {}'.format(window.shape))
                #print('window id: {}'.format(window_id))
                #[x1,y1,x2,y2]
                x1,y1,x2,y2 = x, y, x + window_size[0], y + window_size[1]
                rgb_frames.append(window)
                gt_w = gt_m[y1:y2, x1:x2]
                assert gt_w.shape==window.shape[:2]
                gt_frames.append(gt_w)
                # show window annotations
                w_name = '{:06d}.png'.format(w_id)
                # SWs.overlay_masks_contours(img_bin=gt_w.copy(), im=window.copy(), out_dir=all_args['out_dir_vis'], imgname=w_name,
                #         dpi=200, box_alpha=0.2, show_class=True)
                # SWs.overlay_masks(img_bin=None, im=window.copy(), out_dir=all_args['out_dir_vis'], imgname=w_name,
                #           dpi=200, box_alpha=0.2,  gt=gt_w.copy(),  show_class=True, show_gt_contours=True)
                w_id+=1

            assert len(rgb_frames)==len(gt_frames)
            labeler.get_all_anotations(rgb_frames, gt_frames, fr)
    labeler.save_anns()