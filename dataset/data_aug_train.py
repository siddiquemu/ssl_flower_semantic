from json.tool import main
import sys
import os

# sys.path.insert(0, '/home/siddique/PANet/rgr-public/Python')
coderoot = os.path.dirname(os.path.realpath(__file__)).split('ssl_flower_semantic')[0] + 'ssl_flower_semantic'
print(f'coderoot:{coderoot}')
sys.path.insert(0, f"{coderoot}")
sys.path.insert(0, f'{coderoot}/rgr-public/Python')

from utils.init_pseudo_labels_dir import delete_all
from utils.sliding_windows import sliding_windows
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.clasp2coco import init_annotations_info_pan, Write_To_Json, Write_Segments_info_pan, \
    Write_ImagesInfo, Write_AnnotationInfo, load_clasp_json, get_frame_anns, define_dataset_dictionary_pan
import pycocotools.mask as mask_util
from skimage import measure

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
import argparse

class Proposal_Box(object):
    def __init__(self, out_dir, colorset=None, color_jitter=False, rotation_aug=False,
                 img_list=None, isLabeled=False, save_mask=False, data_type='Peach', vis=False):
        self.imglist = img_list
        self.data_type = data_type
        self.img_mask_labels = None
        self.img_bin = None
        self.img = None
        self.isLabeled = isLabeled
        self.dataset_flower_ins = define_dataset_dictionary_pan(mode='instance')
        self.dataset_flower_pan = define_dataset_dictionary_pan(mode='panoptic')
        self.vis = vis
        self.save_mask = save_mask
        self.out_dir = out_dir
        self.ann_id_count = 1
        if self.isLabeled:
            self.img_id_count = 1
        else:
            self.img_id_count = 46509
        self.verbose = False
        self.apply_color_jitter = color_jitter
        self.jitted_imgs = None
        self.rotation_aug = rotation_aug
        if self.isLabeled:
            if self.apply_color_jitter or self.rotation_aug:
                self.out_img_path = f'{self.out_dir}/trainFlowerAug'
            else:
                self.out_img_path = f'{self.out_dir}/trainFlower'
        else:
            self.out_img_path = f'{self.out_dir}/unlabeledFlower{self.data_type}'

        if not os.path.exists(self.out_img_path):
            os.makedirs(self.out_img_path)
        if self.isLabeled:
            self.instance_label_path = f'{self.out_dir}/semantic_labels'
            self.panoptic_label_path = f'{self.out_dir}/panoptic_labels'
            if not os.path.exists(self.instance_label_path):
                os.makedirs(self.instance_label_path)
                os.makedirs(self.panoptic_label_path)
        self.rgb_mask = None
        self.ranges = [ [78, 96], [258, 276], [342, 354]] #[168, 192] [6, 24]
        self.colorset = colorset

    @staticmethod
    def get_random_angles(ranges=None, factor=None):
        angleSet = [0, 180]
        # angle = 180 will be applied during training
        for intvl in ranges:
            angleSet += random.sample(range(intvl[0], intvl[1]), factor)
        return list(set(angleSet))

    # %%function for converting contours format to segmentation format
    @staticmethod
    def cnt2seg(contours):
        segmentation = [x.flatten() for x in contours]
        segmentation = [[float(y) for y in x] for x in segmentation if len(x) >= 6]
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
        color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(num_of_colors)]
        return color

    def gen_unique_color(self):
        unique = False
        while not unique:
            color = tuple(np.random.choice(range(256), size=3))
            if not (color in self.colorset):
                unique = True
        return color

    def save_semantic_mask(self, imgname, theta):
        """It maps all thing (flower in instance json) categories to class 0, and maps all unlabeled pixels to class 255.
        It maps all stuff categories to contiguous ids starting from 1.
        """
        if theta > 0:
            img_bin = copy.deepcopy(self.img_bin_rot)
        else:
            img_bin = copy.deepcopy(self.img_bin)

        # if len(img_bin.shape)==3:
        #     img_bin=img_bin[:,:,0]
        print(f'unique semantic pixels: {np.unique(img_bin)}')
        smask = np.zeros(img_bin.shape)
        # assign thing and stuff class (unlabeled pixels will be assigned as 255)
        smask[img_bin == 0] = 1  # assign background ((stuff class)) pixels
        smask[img_bin > 0] = 0  # assign flower (thing class) pixels

        smask = smask.astype(np.uint8)
        print(f'semantic ids: {np.unique(smask)[::-1]}')
        # assert len(np.unique(smask))==2, f"thing (0:flower) and stuff (1:background) class ids {np.unique(smask)}"
        pil_img = Image.fromarray(smask).convert('L')
        pil_img.save(f'{self.instance_label_path}/{imgname}')

    def save_rgb_mask(self, imgname):
        """Useful to learn a panoptic model
        self.img_bin and self.img_rot is already augmended by rotation
        """
        cv2.imwrite(f'{self.panoptic_label_path}/{imgname}', self.img_mask_rgb)

    def get_label_masks(self, img_bin):
        # useful for panoptic segmentation mask and bounding boxes from a binary mask'
        self.img_mask_labels = label(img_bin)
        if self.vis:
            self.show_img(self.img_bin, self.img_mask_labels)
        return self.img_mask_labels

    def get_label_boxes(self, label_mask):
        self.fr_boxs = []
        self.fr_masks = []
        self.areas = []
        h, w = label_mask.shape
        # pdb.set_trace()
        # cv2 version 3.x
        _, contours_all_buf, hier_buf = cv2.findContours(label_mask, cv2.RETR_CCOMP,
                                                         cv2.CHAIN_APPROX_NONE)  ##internal and external contours
        if len(contours_all_buf) > 0:
            contours_all = []
            hier = []
            for i in range(len(contours_all_buf)):
                if len(contours_all_buf[i]) >= 6:
                    contours_all.append(contours_all_buf[i])
                    hier.append(hier_buf[0, i, :])

            hier = [hier]
            segmentation_all = self.cnt2seg(contours_all)
            contours_inner = []
            for i in range(len(segmentation_all)):
                exists_parent = (hier[0][i][3] != -1)
                empty_mask = np.zeros((h, w), dtype='uint8')
                if not exists_parent and cv2.contourArea(contours_all[i]) > 100:
                    self.fr_masks.append(segmentation_all[i])
                    self.areas.append(cv2.contourArea(contours_all[i]))
                    self.fr_boxs.append(cv2.boundingRect(contours_all[i]))  #
                else:
                    contours_inner.append(contours_all[i])

    def get_label_boxes_conn_comp(self, theta):
        """Use this function to save augmented anns, semantic, and panoptic mask for both theta=0 or theta>0
        -label_mask is already augmented by rotation
        -self.img_rot is also initiated
        """
        self.fr_boxs = []
        self.fr_masks = []
        self.areas = []
        self.segm_ids = []
        if theta > 0:
            label_mask = copy.deepcopy(self.img_bin_rot)
        else:
            label_mask = copy.deepcopy(self.img_bin)

        h, w = label_mask.shape
        assert self.img_rot.shape[:2] == label_mask.shape, f'{self.img_rot.shape[::2]}=={label_mask.shape}'

        label_mask = label(label_mask > 0)
        img_props_ind = list(np.unique(label_mask))
        print(f'total flower found {len(img_props_ind)}')

        # color = self.get_random_color(len(np.unique(label_mask)))
        self.img_mask_rgb = np.zeros(self.img_rot.shape, dtype='uint8')

        for i in img_props_ind[::-1]:
            color = self.gen_unique_color()
            self.colorset.add(color)
            self.img_mask_rgb[label_mask == i, 0] = color[0]
            self.img_mask_rgb[label_mask == i, 1] = color[1]
            self.img_mask_rgb[label_mask == i, 2] = color[2]
            if i > 0:
                empty_mask = np.zeros(label_mask.shape, dtype='uint8')
                empty_mask[label_mask == i] = 255
                rle = mask_util.encode(np.asfortranarray(empty_mask))

                if mask_util.area(rle) > 200:
                    self.fr_masks.append(rle)
                    # print(f'{i}: Bounding box = {mask_util.toBbox(rle)}')
                    self.fr_boxs.append(mask_util.toBbox(rle))
                    self.areas.append(mask_util.area(rle))
                    self.segm_ids.append(int(color[0] + 256 * color[1] + 256 * 256 * color[2]))
            else:  # background for panoptic
                self.fr_masks.insert(0, [])
                self.areas.insert(0, w * h - sum(self.areas))
                self.fr_boxs.insert(0, [0, 0, w, h])
                self.segm_ids.insert(0, int(color[0] + 256 * color[1] + 256 * 256 * color[2]))

    def get_annotation(self, mask, image=None):
        self.fr_boxs = []
        self.fr_masks = []
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
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
            # print(f'use only original image')

    def get_rotated_labels(self, img, angle=None):
        """fr_masks are in rle format
        self.img_bin is already initiated
        """
        label_mask = copy.deepcopy(self.img_bin)
        label_mask_rot = imutils.rotate_bound(label_mask, angle)
        label_mask_rot[label_mask_rot > 0] = 255
        # used to save sem and pan mask
        self.img_bin_rot = copy.deepcopy(label_mask_rot)
        assert img.shape[:2] == label_mask_rot.shape, f'{img.shape[:2]}=={label_mask_rot.shape}'
        # update fr_boxs and fr_masks
        self.get_label_boxes_conn_comp(angle)

    def update_anns(self):
        # TODO: in case of scale augmentation do not use same anns for all augmented frames
        if self.rotation_aug:
            self.angle_set = self.get_random_angles(ranges=self.ranges, factor=1)

            print(f'apply rotation aug: {self.angle_set}')
        else:
            self.angle_set = [0]

        for angle in self.angle_set:
            if angle > 0:
                print(f'rotated by {angle}')
                # rotate image: self.img will be rotated inplace
                self.img_rot = imutils.rotate_bound(self.img.copy(), angle)
                # rotate anns: mask is in rle format
                # self.fr_boxs, self.fr_masks will be updated inplace for each orientation
                self.get_rotated_labels(self.img_rot, angle)
            else:
                self.img_rot = self.img.copy()
                self.get_label_boxes_conn_comp(angle)

            # save image info
            imgIdnew = 10000 * int('%06d' % self.img_id_count)
            imgname = '{:08d}.png'.format(imgIdnew)
            self.img_write_path = f'{self.out_img_path}/{imgname}'
            print(f'Writing image {imgname}, #anns: {len(self.fr_boxs)}')
            cv2.imwrite(self.img_write_path, self.img_rot)

            if self.save_mask and self.isLabeled:
                self.save_semantic_mask(imgname, angle)
                self.save_rgb_mask(imgname)
            self.dataset_flower_ins = Write_ImagesInfo(self.img_rot, imgname, int(imgIdnew), self.dataset_flower_ins)
            self.dataset_flower_pan = Write_ImagesInfo(self.img_rot, imgname, int(imgIdnew), self.dataset_flower_pan)
            self.img_id_count += 1
            if self.isLabeled:
                segment_infos = []
                for ib, box in enumerate(self.fr_boxs):
                    bboxfinal = [round(x, 2) for x in box[0:4]]
                    # segmPolys = self.fr_masks[ib]
                    mask = self.fr_masks[ib]
                    # area = mask_util.area(mask)#rle area
                    area = self.areas[ib]
                    score = 1
                    if ib > 0:
                        catID = 1
                        annID = self.segm_ids[ib]  # 1000 * int('%06d' % (self.ann_id_count))
                        self.ann_id_count += 1

                        segmPolys = []  # mask['counts'].decode("utf-8") #[]

                        bmask = mask_util.decode(mask)
                        contours = measure.find_contours(bmask, 0.5)
                        for contour in contours:
                            contour = np.flip(contour, axis=1)
                            segmentation = contour.ravel().tolist()
                            if len(segmentation) > 0:
                                segmPolys.append(segmentation)
                        assert int(imgIdnew) == int(os.path.basename(self.img_write_path).split('.')[0])
                        # save annotation for for each image
                        if len(segmPolys) > 0:
                            self.dataset_flower_ins = Write_AnnotationInfo(bboxfinal, segmPolys, int(imgIdnew),
                                                                           int(annID), catID, int(area),
                                                                           self.dataset_flower_ins,
                                                                           instance_certainty=score)
                            segment_infos = Write_Segments_info_pan(segment_infos, bboxfinal, int(annID),
                                                                    catID, int(area), instance_certainty=score)
                    else:
                        # background class
                        catID = 0
                        annID = self.segm_ids[ib]
                        segment_infos = Write_Segments_info_pan(segment_infos, bboxfinal, int(annID),
                                                                catID, int(area), instance_certainty=score)

                self.dataset_flower_pan = init_annotations_info_pan(self.dataset_flower_pan, segment_infos,
                                                                    int(imgIdnew), imgname)
                # print(self.dataset_flower_pan)
            else:
                print('only image infos are updated in dataset dict for unlabeled datasets')

    def get_coco_structure(self):
        # iterate over color augmented images
        # anns scale remain unchanged
        for _, jitt_pil_im in enumerate(self.jitted_imgs):
            if self.apply_color_jitter:
                self.img = self.pil2opencv(jitt_pil_im)
            else:
                self.img = jitt_pil_im
            self.update_anns()

    def get_all_anotations(self, rgb_list, gt_list, fr_num, save_mask=False):
        """self.imglist is the list of high resolution images
        """
        for rgb_w, gt_w in zip(rgb_list, gt_list):

            self.get_rgb_img(rgb_w)
            self.img_bin = gt_w.copy()
            if len(np.unique(self.img_bin)) > 1:
                self.get_coco_structure()

                print(f'Frame: {fr_num}, total augmented frames: {self.img_id_count}')

    def save_anns(self):
        # save flower labels for AppleA train set only
        if self.isLabeled:
            Write_To_Json(f'{self.out_dir}/instances_train_2021.json', self.dataset_flower_ins)
            Write_To_Json(f'{self.out_dir}/panoptic_train_2021.json', self.dataset_flower_pan)
        else:
            Write_To_Json(f'{self.out_dir}/instances_unlabeled_2021_{self.data_type}.json', self.dataset_flower_ins)
            Write_To_Json(f'{self.out_dir}/panoptic_unlabeled_2021_{self.data_type}.json', self.dataset_flower_pan)


def get_all_data_dirs(all_args, dataroot):
    """Dirs of all high resolution raw frames
    """
    all_args['all_data'] = {}

    all_args['all_data']['AppleA_train'] = f'{dataroot}/raw_data/imgs/AppleA_train'
    all_args['all_data']['AppleA'] = f'{dataroot}/raw_data/imgs/AppleA'
    all_args['all_data']['AppleB'] = f'{dataroot}/raw_data/imgs/AppleB'
    all_args['all_data']['Pear'] = f'{dataroot}/raw_data/imgs/Pear'
    all_args['all_data']['Peach'] = f'{dataroot}/raw_data/imgs/Peach'
    
    # for d_name, path in all_args['all_data'].items():
    #     assert os.path.exists(path), 'not exist {}: {}'.format(d_name, path)
    return all_args

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument('--dataset', type=str, default='AppleA_train')

    parser.add_argument(
        '--CV',
        help='cross validation run index',
        default=1, type=int)

    return parser.parse_args()

if __name__ == '__main__':
    """This script will read all the labeled and unlabeled high resolution frames
    and run the strided sliding window to get JSON training files
    - Output size: n_theta*n_cropped_frames*n_input_frames
    """
    args = parse_args()
    print('Called with args:')
    print(args)
    
    data_set = args.dataset
    apply_color_jitter = False
    if data_set=='AppleA_train':
        apply_rotation_aug = True
        isLabeled = True
    else:
        apply_rotation_aug = False
        isLabeled = False
        
    apply_sliding_window = True
    
    save_mask = True
    colorset = set()


    data_root = f'{coderoot}/dataset'
    all_args = {}
    all_args['out_dir'] = f'{data_root}/ssl_train'
    if not os.path.exists(all_args['out_dir']):
        os.makedirs(all_args['out_dir'])
    all_args['out_dir_vis'] = f"{all_args['out_dir']}/vis_anns"
    all_args = get_all_data_dirs(all_args, data_root)

    labeler = Proposal_Box(out_dir=all_args['out_dir'], colorset=colorset, rotation_aug=apply_rotation_aug,
                           color_jitter=apply_color_jitter, isLabeled=isLabeled, data_type=data_set,
                           save_mask=save_mask, vis=False)

    for data_type in [data_set]:  # , 'AppleB', 'Peach', 'Pear']: #['AppleA_train']:#,
        data_dir = all_args['all_data'][data_type]
        folder = glob.glob(data_dir + '/*')
        folder.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        w_id = 1
        
        for img_path in folder:
            rgb_frames = []
            gt_frames = []
            img = cv2.imread(img_path)

            if data_type in ['AppleA', 'AppleA_train']:
                fr = float(os.path.basename(img_path).split('.')[0].split('IMG_')[-1])
                step_size = [img.shape[1] // 16, img.shape[0] // 16]  # [step_w, step_h] #8.6 or 16,12 or 16, 16
                window_size = [img.shape[1] // 8, img.shape[0] // 8]  # [w_W, w_H] #4,3 or 8,6, or 8,8

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
            gt_m = SWs.get_gt_mask(fr, data_type, dataroot=data_root)


            print(f'input shape: {img.shape}, gt_m shape: {gt_m.shape}')
            assert img.shape[:2]==gt_m.shape

            for (x1, y1, x2, y2, window) in SWs.sliding_window():
                if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                    pdb.set_trace()

                rgb_frames.append(window)
                gt_w = gt_m[y1:y2, x1:x2]
                assert gt_w.shape == window.shape[:2]
                gt_frames.append(gt_w)
                w_name = '{:06d}.png'.format(w_id)
                w_id += 1

            assert len(rgb_frames) == len(gt_frames)
            labeler.get_all_anotations(rgb_frames, gt_frames, fr)
            
    labeler.save_anns()