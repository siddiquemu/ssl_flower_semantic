# Setup detectron2 logger
from time import process_time
import sys
#sys.path.insert(0, '/home/siddique/miniconda3/envs/det2/lib/python3.7/site-packages/detectron2')
import detectron2
import torch
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import glob
import imutils
import pdb
from collections import OrderedDict

# import some common detectron2 utilities
from detectron2.structures import Instances, Boxes
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.utils.visualizer import _PanopticPrediction
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model, build_proposal_generator, build_roi_heads, \
    build_box_head
from detectron2.modeling.postprocessing import sem_seg_postprocess, detector_postprocess
from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs, PanopticFPN

from detectron2.data.datasets import register_coco_instances, register_coco_panoptic_separated
from detectron2.utils.visualizer import ColorMode
import PIL
from PIL import Image, ImageDraw
#from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
#from json_gen_outers_polygon import *
#basic torch operation
#https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/
from utils import *
import torch.nn.functional as F

class panoptic_fpn_flower(object):

    def __init__(self, init_params):
        self.instance_class = 1
        self.sem_class = 2
        self.data = init_params['data']
        self.pred_score = init_params['pred_score']
        self.nms_thr = init_params['nms_thr']
        self.gpu_id = init_params['gpu_id']
        self.iter_i = init_params['iter_i']
        self.pretrained = init_params['pretrained']
        self.default_predictor = init_params['default_predictor']
        self.test_aug = init_params['test_aug']
        self.angle_ranges = init_params['angle_ranges']
        self.data_already_registered = init_params['data_already_registered']
        #TODO
        self.other_aug_param = {}

        self.img_dir = init_params['unlabeled_img_dir']
        self.panoptic_masks = init_params['panoptic_masks']
        self.panoptic_json = init_params['panoptic_json']
        self.sem_seg_masks = init_params['sem_seg_masks']
        self.instance_json = init_params['instance_json']
        self.model_path = init_params['model_path']
        self.combine_panoptic = init_params['combine_panoptic']

        self.combine_overlap_thresh = 0.5
        self.combine_stuff_area_thresh = 5
        self.combine_instances_score_thresh = 0.5
        #default config
        self.cfg = get_cfg()
        self.aug = T.ResizeShortestEdge(
                [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
            )

        #super(panoptic_fpn_flower, self).__init__()
        


    @staticmethod
    def get_random_angles(ranges=None, factor=None):
        angleSet = [0]
        for intvl in ranges:
            angleSet += random.sample(range(intvl[0], intvl[1]), factor)
        return angleSet

    def get_test_config(self):
        if self.test_aug:
            self.angleSet = self.get_random_angles(ranges=self.angle_ranges, factor=4)
        else:
            self.angleSet = [0]

        if self.pretrained:
            if self.test_aug:
                self.class_id = 9
            else:
                self.class_id = [9]
        else:
            if self.test_aug:
                self.class_id = 0
            else:
                #only inference
                self.class_id = [0, 1] # 0:flower, 1: background


        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        
        if self.pretrained:
            self.flower_metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.pred_score
            self.cfg.MODEL.PANOPTIC_FPN.TUFF_AREA_LIMIT = 5#4096 #default:4096
        else:
            if not self.data_already_registered:
                register_coco_panoptic_separated(name="AppleA_train",
                                                metadata={},
                                                image_root=self.img_dir,
                                                panoptic_root=self.panoptic_masks,
                                                panoptic_json=self.panoptic_json,
                                                sem_seg_root=self.sem_seg_masks,
                                                instances_json=self.instance_json)
            else:
                print(f'sdata_already_registered:{self.data_already_registered}')
            for d in ['train']:
                #DatasetCatalog.register("AppleA_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
                MetadataCatalog.get("AppleA_" + d).set(thing_classes=["flower"],
                                                    stuff_classes=["flower", "background"])
            self.flower_metadata = MetadataCatalog.get("AppleA_train")


            self.cfg.MODEL.DEVICE = self.gpu_id
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.nms_thr
            self.cfg.DATASETS.TEST = ("AppleA_split",)
            #cfg.MODEL.BACKBONE.FREEZE_AT = 4 #no freeze during test
            # Inference with a panoptic segmentation model
            #self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.1] #between proposals to be pred
            
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.nms_thr #default:0.8
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.pred_score # used as reg_score_thr also
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.instance_class
            self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = self.sem_class
            self.cfg.MODEL.PANOPTIC_FPN.TUFF_AREA_LIMIT = 5#4096 #default:4096
            self.cfg.MODEL.PANOPTIC_FPN.INSTANCES_CONFIDENCE_THRESH = 0.1 #default: 0.5
            self.cfg.MODEL.PANOPTIC_FPN.OVERLAP_THRESH = 0.5 #default: 0.5 ??

            #cfg.MODEL.WEIGHTS = '/home/abubakarsiddique/trainPanopticModel/flower_model/model_final.pth'
            self.cfg.MODEL.WEIGHTS = self.model_path
        print(f'>>> get model from {self.cfg.MODEL.WEIGHTS}')


    def init_detector(self):
        #prepare configuration file
        self.get_test_config()
        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        #TODO: configure test model to get instance, semantic and panoptic outputs separately
        if self.default_predictor:
            self.model = DefaultPredictor(self.cfg)
        else:
            # build model
            self.cfg = self.cfg.clone()  # cfg can be modified by model
            self.model = build_model(self.cfg)
            self.model.eval()
            if len(self.cfg.DATASETS.TEST):
                self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
            
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(self.cfg.MODEL.WEIGHTS)
            print(f'>>> load model from {self.cfg.MODEL.WEIGHTS}')
            self.input_format = self.cfg.INPUT.FORMAT

    def transform_raw_proposals(self, pre_proposals,batched_input, image_size):
        """Desired output resolution is the network input size (800, 900)
        pre_proposals are in actual input image resolution but need to map to network input size
        """
        #actual input sizes: sizes for proposals
        height = batched_input.get("height", image_size[0])
        width = batched_input.get("width", image_size[1])
        #network raw output size
        output_height_tmp = pre_proposals.__dict__['_image_size'][0]
        output_width_tmp = pre_proposals.__dict__['_image_size'][1]
        #print(f'network size:{output_height_tmp},{output_width_tmp} , input image size:{height},{width}')
        scale_x, scale_y = (
            output_width_tmp / width,
            output_height_tmp / height,
        )

        pre_proposals.proposal_boxes.scale(scale_x, scale_y)
        pre_proposals.proposal_boxes.clip(pre_proposals.image_size)
        return pre_proposals

   

    def predict_single(self, img, pre_proposals=None, do_postprocess=True, apply_jitter=False, return_coarse=False):
        """
        Args:
            img (np.ndarray): an image of shape (H, W, C) (in BGR order).
        """
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            img = img[:, :, ::-1]
        if self.default_predictor:
            processed_results = self.model(img)
        else:
            processed_results = []
            with torch.no_grad():
                #process single image in the batch list
                #print('can regress the proposals')
                height, width = img.shape[:2]
                #transformed inputs for model
                image = self.aug.get_transform(img).apply_image(img)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs = {"image": image, "height": height, "width": width}
                batched_inputs = [inputs]

                images = self.model.preprocess_image(batched_inputs)
                features = self.model.backbone(images.tensor)
                sem_seg_results, _ = self.model.sem_seg_head(features, None)
                #images.tensor.shape
                #TODO: proposals willbe replaced by clusters
                #proposals[0].get_fields()['proposal_boxes'][0]
                #cfg.MODEL.PROPOSAL_GENERATOR.NAME
                if pre_proposals is not None:
                    # self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.0 #default:0.8
                    # self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0 #reg_score_thr

                    proposals, _ = self.model.proposal_generator(images, features, None)
                    # pred_instances, _ = self.box_predictor.inference(predictions, proposals)
                    # pred_instances = self.forward_with_given_boxes(features, pred_instances)
                    
                    # def forward_with_given_boxes(self, features, instances):
                    #     """
                    #     Use the given boxes in `instances` to produce other (non-box) per-ROI outputs such as instance masks.

                    #     Args:
                    #         features: same as in `forward()`
                    #         instances (list[Instances]): instances to predict other outputs. Expect the keys
                    #             "pred_boxes" and "pred_classes" to exist.

                    #     Returns:
                    #         instances (Instances):
                    #             the same `Instances` object, with extra
                    #             fields such as `pred_masks` or `pred_keypoints`.
                    #     """
                    #     assert not self.training
                    #     assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

                    #     if self.mask_on:
                    #         features = [features[f] for f in self.in_features]
                    #         x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
                    #         return self.mask_head(x, instances)
                    #     else:
                    #         return instances
                    
                    #TODO: use precomputed proposals (pseudo labels) to regress using rcnn head
                    #update proposals using pre-proposals
                    #proposals[0].proposal_boxes.tensor.cpu().numpy()
                    #proposals[0].objectness_logits.cpu().numpy().shape
                    if len(pre_proposals)>0:
                        #will be used for jitter application
                        # self.model.roi_heads.box_predictor.test_score_thresh = 0.05
                        # self.model.roi_heads.box_predictor.test_nms_thresh = 0.6
                        self.model.roi_heads.box_predictor.test_topk_per_image = 500

                        updt_proposals = proposals[0][0:len(pre_proposals)]
                        #xywh to x1y1x2y2
                        #print(updt_proposals)
                        #print(pre_proposals.shape)
                        pre_proposals[:,2:4] = pre_proposals[:,0:2] + pre_proposals[:,2:4]
                        updt_proposals.proposal_boxes.tensor = torch.tensor(pre_proposals).to(self.gpu_id)
                        updt_proposals = self.transform_raw_proposals(updt_proposals, batched_inputs[0], images.image_sizes[0])
                        
                        #TODO: sub-model results from class StandardROIHeads(ROIHeads): in roi_heads.py
                        features_roi = [features[f] for f in self.model.roi_heads.box_in_features]
                        box_features = self.model.roi_heads.box_pooler(features_roi, [updt_proposals.proposal_boxes])
                        box_features = self.model.roi_heads.box_head(box_features)
                        class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)
                        del box_features
                        
                        if not apply_jitter:
                            #process logits and regression
                            pred_boxes = self.model.roi_heads.box_predictor.box2box_transform.apply_deltas(box_regression, 
                                                updt_proposals.proposal_boxes.tensor)
                            pred_scores = F.softmax(class_logits, -1)
                            # create labels for each prediction
                            pred_labels = torch.arange(self.cfg.MODEL.ROI_HEADS.NUM_CLASSES, device=self.gpu_id)
                            pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)
                            pred_labels = pred_labels[:, 1:].squeeze(dim=1).detach()
                            pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
                            #confident_detections = box_results[box_results.scores > 0.9]
                            # from fast_rcnn.py
                            # def fast_rcnn_inference_single_image(
                            #         boxes,
                            #         scores,
                            #         image_shape: Tuple[int, int],
                            #         score_thresh: float,
                            #         nms_thresh: float,
                            #         topk_per_image: int,
                            #     ):
                            # Convert to Boxes to use the `clip` function ...
                            #wo nms and score filter
                            pred_boxes = Boxes(pred_boxes)
                            pred_boxes.clip(updt_proposals._image_size)
                            #TODO: apply nms and score fltering
                            results = {'pred_boxes':pred_boxes, 'scores':pred_scores, 'pred_classes':pred_labels}
                            box_results = Instances(updt_proposals._image_size, **results)
                            #predict mask
                            #pdb.set_trace()
                            detector_results = self.model.roi_heads.forward_with_given_boxes(features, [box_results])
                        else:
                            #apply NMS and pred_scores when use jittered boxes
                            detector_results, _ = self.model.roi_heads.box_predictor.inference((class_logits, box_regression), [updt_proposals])
                            detector_results = self.model.roi_heads.forward_with_given_boxes(features, detector_results)
                        #original resolution
                        height = batched_inputs[0].get("height", images.image_sizes[0][0])
                        width = batched_inputs[0].get("width", images.image_sizes[0][1])
                        #print(f'raw input size: {height,width}')
                        #map boxes to original resolution
                        detector_r = detector_postprocess(detector_results[0], height, width, mask_threshold=0.5)

                        #return regressed box and the corresponding 28x28 coarse mask
                        if return_coarse:
                            detector_r.pred_masks = detector_results[0].pred_masks
                        
                    else:
                        detector_r=None
                    
                    return detector_r
                else:
                    proposals, _ = self.model.proposal_generator(images, features, None)
                    
                
                detector_results, _ = self.model.roi_heads(images, features, proposals, None)

            if do_postprocess:
                processed_results = []
                for sem_seg_result, detector_result, input_per_image, image_size in zip(
                    sem_seg_results, detector_results, batched_inputs, images.image_sizes
                ):
                    #actual input sizes
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
                    detector_r = detector_postprocess(detector_result, height, width)
                    #return 28x28 scoremap
                    detector_r.pred_masks = detector_result.pred_masks
                    processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})
                    if self.combine_panoptic:
                        #torch.Size([2, 1152, 1296]): argmax(dim=0) assign each pixel to either 0: flower or 1: background
                        # based on classwise soft prediction values: need to apply softmax to get normalized scoremap
                        panoptic_r = combine_semantic_and_instance_outputs(
                            detector_r,
                            sem_seg_r.argmax(dim=0),
                            self.combine_overlap_thresh,
                            self.combine_stuff_area_thresh,
                            self.combine_instances_score_thresh,
                        )
                        processed_results[-1]["panoptic_seg"] = panoptic_r
                    else:
                        processed_results = processed_results[0]


        return processed_results

    def predict_batch(self, batched_inputs):
        pass