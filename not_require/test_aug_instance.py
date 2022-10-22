# Setup detectron2 logger
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

# import some common detectron2 utilities
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
from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs

from detectron2.data.datasets import register_coco_instances, register_coco_panoptic_separated
from detectron2.utils.visualizer import ColorMode
import PIL
from PIL import Image, ImageDraw
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
#from json_gen_outers_polygon import *
#basic torch operation
#https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/
from utils import *


#TODO: implement panoptic format of test results for evaluation
if __name__ == '__main__':
    # This script should use only for inference
    # For test-time-augmentation we will use separate script
    #instance segmentation: 0: flower
    #semantic segmentation: 0:, 1:
    dataset = 'Peach_test'
    NAS = '/media/NAS/LabFiles/Walden'
    det2 = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/Detectron2'

    if dataset=='AppleA_train':
        train_dir = det2 + '/flower_dataset/AppleA_Panoptic/DataFormattedProperly/SplitImages4x3/train/'

        NAS = '/media/NAS/LabFiles/Walden'
        base_dir = NAS + '/trainTestSplit/train/dataFormattedProperly/splitImages4x3'

        img_dir = base_dir + '/flowersSplit'
        panoptic_masks = base_dir + '/rgbMasksSplit'
        panoptic_json = base_dir + '/appleA_panoptic_split4x3_train.json'

        # 0: flower, 255: background
        sem_seg_masks = base_dir + '/semMasksSplit'
        instance_json = base_dir + '/appleA_instance_split4x3_train.json'

    if dataset=='AppleA_test':
        data = 'AppleA'
        base_dir = NAS + '/trainTestSplit/test/dataFormattedProperly/splitImages4x3'

        img_dir = base_dir + '/flowersSplit'
        panoptic_masks = base_dir + '/rgbMasksSplit'
        panoptic_json = base_dir + '/appleA_panoptic_split4x3_test.json'

        # 0: flower, 255: background
        sem_seg_masks = base_dir + '/semMasksSplit'
        instance_json = base_dir + '/appleA_instance_split4x3_test.json'
        img_fmt = '.jpg'

    if dataset=='AppleB_test':
        data = 'AppleB'
        base_dir = NAS + '/otherFlowerDatasets/AppleB/dataFormattedProperly/splitImages4x3'

        img_dir = base_dir + '/flowersSplit'
        panoptic_masks = base_dir + '/rgbMasksSplit'
        panoptic_json = base_dir + '/AppleB_panoptic_split4x3_test.json'

        # 0: flower, 255: background
        sem_seg_masks = base_dir + '/semMasksSplit'
        instance_json = base_dir + '/AppleB_instance_split4x3_test.json'

        img_fmt = '.bmp'

    if dataset=='Peach_test':
        data = 'Peach'
        base_dir = NAS + '/otherFlowerDatasets/Peach/dataFormattedProperly/splitImages4x3'

        img_dir = base_dir + '/flowersSplit'
        panoptic_masks = base_dir + '/rgbMasksSplit'
        panoptic_json = base_dir + '/Peach_panoptic_split4x3_test.json'

        # 0: flower, 255: background
        sem_seg_masks = base_dir + '/semMasksSplit'
        instance_json = base_dir + '/Peach_instance_split4x3_test.json'

        img_fmt = '.bmp'

    if dataset=='Pear_test':
        data = 'Pear'
        base_dir = NAS + '/otherFlowerDatasets/Pear/dataFormattedProperly/splitImages4x3'

        img_dir = base_dir + '/flowersSplit'
        panoptic_masks = base_dir + '/rgbMasksSplit'
        panoptic_json = base_dir + '/Pear_panoptic_split4x3_test.json'

        # 0: flower, 255: background
        sem_seg_masks = base_dir + '/semMasksSplit'
        instance_json = base_dir + '/Pear_instance_split4x3_test.json'

        img_fmt = '.bmp'


    pretrained = 0

    regress_cluster= 0
    test_aug = 1
    save_aug = 0
    write_img = True
    save_infer_panoptic_json = 1
    eval_only = 0
    show_demo = 1
    regress_model_type = 'instance_segmentation'

    #for eval results
    #-----------------------------------------------------------------------
    outpath=os.path.join('/home/abubakarsiddique/trainPanopticModel', dataset)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    else:
        delete_all(outpath, fmt='jpg')

    contour_area_thr = 0
    rgbmask_dir = os.path.join(outpath, 'rgbMasksSplit/')
    if not os.path.exists(rgbmask_dir):
        os.makedirs(rgbmask_dir)

    sem_mask_dir = os.path.join(outpath, 'semMasksSplit/')
    if not os.path.exists(sem_mask_dir):
        os.makedirs(sem_mask_dir)
    if not eval_only:
        #global variable to keep pan anoos for every frame
        pan_anns = []
        inst_anns = []
        imgs = []
        colorsdict = set()
        #-----------------------------------------------------------------------
        #prepare configuration file
        cfg, angleSet, class_id, flower_metadata = get_test_config(pretrained=pretrained, test_aug=test_aug, img_dir=img_dir, panoptic_masks=panoptic_masks,
                     panoptic_json=panoptic_masks, sem_seg_masks=sem_seg_masks, instance_json=instance_json)

        #TODO: configure test model to get instance, semantic and panoptic outputs separately
        predictor = DefaultPredictor(cfg)

        if show_demo:
            cv2.namedWindow('image')
            cv2.resizeWindow('image', 1920, 1080)

        #images = sorted(glob.glob(storage + 'tracking_wo_bnw/data/CLASP1/train_gt/' + cam + '/img1/*.png'))
        #images = sorted(glob.glob(storage + 'Detectron2/flower_dataset/AppleA_renamed/FlowerImages/*.jpg'))
        images = sorted(glob.glob(base_dir + '/flowersSplit/*'+img_fmt))
        for im_name in images:
            fr_num = float(os.path.basename(im_name).split('.')[0])

            print('Data {} Frame {}'.format(dataset, os.path.basename(im_name)))
            im = cv2.imread(im_name)
            #im = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2))
            patch_scale = 1
            #TODO: For end-to-end test we will use 12 splitted patch at a time and then combine to get full resolution output
            #TODO: How to combine the splitted patch predictions? : May be we will use semantic predictions only for evaluation and merge the instance
            # predictions where semantic predictions fail.
            patches = split_patches(im=im, even_scale=1)

            final_mask = []
            for angle in angleSet:
                segms = []
                segms_info = []
                combine_cat_id = {}
                map_cat_id = {}

                img_mask = torch.empty((im.shape[0],im.shape[1]), dtype=torch.uint8, device = 'cuda')
                rgb_mask = torch.empty((im.shape[0],im.shape[1]), dtype=torch.uint8, device = 'cuda')
                for imgrot in patches:
                    if angle != 0:
                        print('Frame {} Rotated by {}'.format(os.path.basename(im_name), angle))
                        imgrot = imutils.rotate_bound(imgrot, angle)
                        #imgrot = imutils.rotate_bound(im, angle)
                    else:
                        imgrot = imgrot

                    #TODO: use instance and semantic predictions separately to find the final pred from the augmmented set
                    # After getting the augmentation benefits from both task, combine using panoptic methods
                    out = predictor(imgrot)
                    #TODO: test aug on semantics
                    # 0: flower, 1: background > binary mask
                    sem_seg = out['sem_seg']
                    #TODO: test aug on instance
                    # 0: flower > box, mask
                    instances = out['instances']

                    if show_demo:
                        panoptic_seg, segments_info = out['panoptic_seg']
                        imgrot = save_augmented_image(imgrot, flower_metadata, panoptic_seg,
                                                      segments_info, outpath, im_name, angle, class_id, write_img=write_img)

                        cv2.imshow('image', imgrot)
                        cv2.waitKey(500)
                    #unifor patch infos
                    #combine_cat_id, map_cat_id might be empty due to background only
                    combine_cat_id, map_cat_id, panoptic_seg = combine_uniform_patch_info(combine_cat_id,
                                                                                          map_cat_id,
                                                                                          panoptic_seg,
                                                                                          segments_info,
                                                                                          class_id=class_id)
                    # remap seg for flower class
                    panoptic_seg = mask_image_remap(panoptic_seg, im, angle, even_scale=patch_scale)
                    segms.append(panoptic_seg)


            #when splitting and merging happen during test time
            panoptic_seg = merge_segms(segms, even_scale=patch_scale, im_mask=img_mask)
            #check that flower is found in the panoptic pred
            if combine_cat_id:
                img_segms_info = list(combine_cat_id.values())
                if class_id in map_cat_id:
                    #need to inverse the panoptic seg later for vis
                    panoptic_seg[panoptic_seg==map_cat_id[class_id]] = 1 #flower
                    panoptic_seg[panoptic_seg != 1] = 0 # background
                    final_mask.append(panoptic_seg)
            else:
                print('only background found')
                final_mask.append(panoptic_seg)

            if show_demo:
                imgrot = save_augmented_image(imgrot, flower_metadata, panoptic_seg,
                                     segments_info, outpath, im_name, angle, class_id)

                cv2.imshow('image', imgrot)
                cv2.waitKey(5000)

            imgs, pan_anns, \
            inst_anns, colorsdict = save2panopticJSON(int(fr_num),  panoptic_seg,
                                                      segments_info, flower_metadata,
                                                      imgs, pan_anns, inst_anns, colorsdict,
                                                      rgbmask_dir, sem_mask_dir)

            if regress_cluster:
                transform_gen = T.ResizeShortestEdge(
                    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
                )
                model = build_model(cfg)
                model.eval()
                with torch.no_grad():
                    checkpointer = DetectionCheckpointer(model)
                    checkpointer.load(cfg.MODEL.WEIGHTS)
                    #list of images also can be used
                    if regress_model_type=='instance_segmentation':
                        raw_inputs = {"image": imgrot, "height": imgrot.shape[0], "width": imgrot.shape[1]}
                        #transformed inputs for model
                        image = transform_gen.get_transform(imgrot).apply_image(imgrot)
                        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                        inputs = {"image": image, "height": image.shape[1], "width": image.shape[2]}
                        img = model.preprocess_image([inputs])
                        features = model.backbone(img.tensor)
                        proposals, _ = model.proposal_generator(img, features, None)
                        #images.tensor.shape
                        #TODO: proposals willbe replaced by clusters
                        #proposals[0].get_fields()['proposal_boxes'][0]
                        #cfg.MODEL.PROPOSAL_GENERATOR.NAME
                        results, _ = model.roi_heads(img, features, proposals)
                        outputs = model._postprocess(results, [raw_inputs], img.image_sizes)
                        #outputs = outputs[0]['instances'].get_fields()
                        print(outputs[0]['instances'].get('pred_boxes'))
                        print(outputs[0]['instances'].get('pred_classes'))
                        v = Visualizer(imgrot[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
                        out = v.draw_instance_predictions(outputs[0]['instances'].to('cpu'))
                    else:
                        #process splitted patches in parallel
                        raw_inputs = []
                        net_inputs = []
                        for imgrot in patches:
                            raw_inputs.append({"image": imgrot, "height": imgrot.shape[0], "width": imgrot.shape[1]})
                            #transformed inputs for model
                            image = transform_gen.get_transform(imgrot).apply_image(imgrot)
                            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                            net_inputs.append({"image": image, "height": image.shape[1], "width": image.shape[2]})

                        imgs = model.preprocess_image([inputs])
                        features = model.backbone(imgs.tensor)
                        proposals, _ = model.proposal_generator(imgs, features, None)
                        #images.tensor.shape
                        #TODO: proposals willbe replaced by clusters
                        #proposals[0].get_fields()['proposal_boxes'][0]
                        #cfg.MODEL.PROPOSAL_GENERATOR.NAME
                        results, _ = model.roi_heads(imgs, features, proposals)
                        outputs = model._postprocess(results, raw_inputs, imgs[0].image_sizes)

                        #combined segms results for flower class
                        panoptic_seg, segments_info = merge_segms(outputs=outputs)

                        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                        out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

                cv2.imwrite(os.path.join(outpath, '{}_{}'.format(angle, os.path.basename(im_name))), out.get_image()[:, :, ::-1])
                    #mask_features = [features[f] for f in model.roi_heads.in_features]
                    #mask_features = model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])
            # get possible clusters and instance indexes
            #save_augmented_masks(final_mask, angleSet, outpath, im_name)

            final_masks = torch.cat(([mask.unsqueeze(0).type(torch.float) for mask in final_mask]), dim=0)
            final_masks = final_masks.mean(0)
            final_masks[final_masks >= 0.5] = 255
            final_masks[final_masks < 0.5] = 0
            # during SSL iteration split the rgb image and
            # corresponding mask when flower detection available
            if save_aug:
                cv2.imwrite(os.path.join(outpath, 'mask/{}'.format(os.path.basename(im_name))),
                            final_masks.cpu().numpy())
            if len(img_segms_info) == 1:
                print('segms_info: {}, map_cat_id: {}'.format(img_segms_info, map_cat_id))
                assert img_segms_info[0]['id'] == map_cat_id[class_id]
                rgb_mask[final_masks > 0] = map_cat_id[class_id] #img_segms_info[0]['id']  # map_cat_id[9]
            v = Visualizer(im[:, :, ::-1],
                           metadata=flower_metadata,
                           scale=0.5,)
                           #instance_mode=ColorMode.IMAGE_BW)
            out = v.draw_panoptic_seg_predictions(rgb_mask.to("cpu"), img_segms_info)
            #if show_demo:
                #cv2.imshow('image', out.get_image()[:, :, ::-1])
                #cv2.waitKey(300)
            if write_img:
                cv2.imwrite(os.path.join(outpath, '{}'.format(os.path.basename(im_name))),
                        out.get_image()[:, :, ::-1])
        if show_demo:
            cv2.destroyAllWindows()

        if save_infer_panoptic_json:
            cats = define_categories_dictionary()

            panopticjson = {}
            panopticjson['images'] = imgs
            panopticjson['annotations'] = pan_anns
            panopticjson['categories'] = cats

            # s=json.dumps(instancejson)
            with open(os.path.join(outpath, dataset+'_panoptic_split4x3_inference.json'), 'w') as f:
                # f.write(s)
                json.dump(panopticjson, f, cls=NpEncoder)
                # json.dump(data, fp,cls=NpEncoder)

            # instancejson={}
            # instancejson['images']=imgs
            # instancejson['annotations']=inst_anns
            # instancejson['categories']=cats

            # s=json.dumps(instancejson)
            # with open(base_dir+'/polygon/'+'appleA_instance_split4x3_polygon.json','w') as f:
            # f.write(s)

            # python3 /home/abubakarsiddique/anaconda3/envs/det2/lib/python3.8/site-packages/panopticapi/evaluation.py --gt_json_file /media/NAS/LabFiles/Walden/trainTestSplit/test/dataFormattedProperly/splitImages4x3/appleA_panoptic_split4x3_test.json --pred_json_file /home/abubakarsiddique/trainPanopticModel/AppleA_test/appleA_panoptic_split4x3_inference.json --gt_folder /media/NAS/LabFiles/Walden/trainTestSplit/test/dataFormattedProperly/splitImages4x3/rgbMasksSplit --pred_folder /home/abubakarsiddique/trainPanopticModel/AppleA_test/rgbMasksSplit/

    else:
        # IoU metric
        eval_type = 'macro' #'binary'
        p_masks = sorted(glob.glob(sem_mask_dir +'*.png'))
        g_masks = sorted(glob.glob(base_dir+'/bMasksSplit/*.png'))
        assert len(g_masks) == len(p_masks)
        ious = []
        F1s = []
        Prs = []
        Rclls = []


        for y_pred, y_true in zip(p_masks, g_masks):
            #y_true = base_dir+'/bMasksSplit/{}.png'.format(int(float(os.path.basename(y_pred).split('.')[0])))
            assert float(os.path.basename(y_true).split('.')[0])==float(os.path.basename(y_pred).split('.')[0])
            gt_mask = cv2.imread(y_true, 0)
            #if 255 in gt_mask:
            gt_mask[gt_mask>0] = 1
            if 1 in gt_mask:
                pred_mask = cv2.imread(y_pred,0)
                pred_mask[pred_mask == 0] = 1
                pred_mask[pred_mask == 255] = 0

                jscore = jaccard_score(gt_mask.flatten(), pred_mask.flatten(), average=eval_type)#, pos_label=1
                F1 = f1_score(gt_mask.flatten(), pred_mask.flatten(), average=eval_type)
                Pr = precision_score(gt_mask.flatten(), pred_mask.flatten(), average=eval_type)
                Rcll = recall_score(gt_mask.flatten(), pred_mask.flatten(), average=eval_type)

                print('jscore: {} between gt: {}, pred: {}'.format(jscore, float(os.path.basename(y_true).split('.')[0]), float(os.path.basename(y_pred).split('.')[0])))
                ious.append(jscore)
                F1s.append(F1)
                Prs.append(Pr)
                Rclls.append(Rcll)

        print('IoU: {:.2f} %'.format(np.nanmean(ious)*100))
        print('F1: {:.2f} %'.format(np.nanmean(F1s)*100))
        print('Pr: {:.2f} %'.format(np.nanmean(Prs)*100))
        print('Rcll: {:.2f} %'.format(np.nanmean(Rclls)*100))