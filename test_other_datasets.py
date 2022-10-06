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
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
#from json_gen_outers_polygon import *
#basic torch operation
#https://jhui.github.io/2018/02/09/PyTorch-Basic-operations/

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def mask_image_remap(panoptic_seg, img_org, angle, even_scale=2):
    # to remap coarse segmentation mask
    #actual path scale
    patch_Horg = img_org.shape[0] // even_scale
    patch_Worg = img_org.shape[1] // even_scale
    # TODO: place rotated coarse mask on rotated mask image, re-rotate the mask image and crop the remapped mask
    mask = panoptic_seg.cpu().numpy().astype('float')
    imgrerot = imutils.rotate_bound(mask, -angle)  # mask_image
    Hrot,Wrot = imgrerot.shape[0]//2, imgrerot.shape[1]//2
    H, W = patch_Horg//2, patch_Worg//2
    assert Hrot>=H and Wrot>=W
    mask_org = imgrerot[Hrot - H: Hrot + H,Wrot - W:Wrot + W]
    # save masks at multiple inference
    assert (2*H, 2*W)==mask_org.shape
    return torch.tensor(mask_org.astype('int')).cuda()

def split_patches(im=None, even_scale=None):
    M = im.shape[0] // even_scale
    N = im.shape[1] // even_scale
    return [im[x:x + M, y:y + N] for x in range(0, im.shape[0], M) for y in range(0, im.shape[1], N)]

def merge_segms(segms_list=None, segms_info=None, even_scale=2, im_mask=None):
    merged_info = []
    M = im_mask.shape[0] // even_scale
    N = im_mask.shape[1] // even_scale
    i=0
    for x in range(0, im_mask.shape[0], M):
        for y in range(0, im_mask.shape[1], N):
            im_mask[x:x + M, y:y + N] = segms_list[i]
            i+=1
    return im_mask

def get_batched_inputs(model, original_image, batched_inputs):
    assert isinstance(list, batched_inputs)
    if model.input_format == "RGB":
        # whether the model expects BGR inputs or RGB
        original_image = original_image[:, :, ::-1]
    height, width = original_image.shape[:2]
    image = model.transform_gen.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    return batched_inputs.append({"image": image, "height": height, "width": width})

def predict_batch_panoptic(model, batched_inputs):
    """
    Args:
        batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
            Each item in the list contains the inputs for one image.

            For now, each item in the list is a dict that contains:

            * "image": Tensor, image in (C, H, W) format.
            * "instances": Instances
            * "sem_seg": semantic segmentation ground truth.
            * Other information that's included in the original dicts, such as:
              "height", "width" (int): the output resolution of the model, used in inference.
              See :meth:`postprocess` for details.

    Returns:
        list[dict]:
            each dict is the results for one image. The dict contains the following keys:

            * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
            * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
            * "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
              See the return value of
              :func:`combine_semantic_and_instance_outputs` for its format.
    """
    images = [x["image"].to(self.device) for x in batched_inputs]
    images = [model.normalizer(x) for x in images]
    images = ImageList.from_tensors(images, model.backbone.size_divisibility)
    features = model.backbone(images.tensor)

    if "proposals" in batched_inputs[0]:
        proposals = [x["proposals"].to(model.device) for x in batched_inputs]
        proposal_losses = {}

    if "sem_seg" in batched_inputs[0]:
       gt_sem_seg = [x["sem_seg"].to(model.device) for x in batched_inputs]
       gt_sem_seg = ImageList.from_tensors(
           gt_sem_seg, model.backbone.size_divisibility, model.sem_seg_head.ignore_value
        ).tensor
    else:
        gt_sem_seg = None

    sem_seg_results, sem_seg_losses = model.sem_seg_head(features, gt_sem_seg)

    if "instances" in batched_inputs[0]:
        gt_instances = [x["instances"].to(model.device) for x in batched_inputs]
    else:
        gt_instances = None
    if model.proposal_generator:
        proposals, proposal_losses = model.proposal_generator(images, features, gt_instances)
    detector_results, detector_losses = model.roi_heads(
        images, features, proposals, gt_instances
    )

    if model.training:
        losses = {}
        losses.update(sem_seg_losses)
        losses.update({k: v * model.instance_loss_weight for k, v in detector_losses.items()})
        losses.update(proposal_losses)
        return losses

    processed_results = []
    for sem_seg_result, detector_result, input_per_image, image_size in zip(
            sem_seg_results, detector_results, batched_inputs, images.image_sizes
    ):
        height = input_per_image.get("height", image_size[0])
        width = input_per_image.get("width", image_size[1])
        sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
        detector_r = detector_postprocess(detector_result, height, width)

        processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

        if model.combine_on:
            panoptic_r = combine_semantic_and_instance_outputs(
                detector_r,
                sem_seg_r.argmax(dim=0),
                model.combine_overlap_threshold,
                model.combine_stuff_area_limit,
                model.combine_instances_confidence_threshold,
            )
            processed_results[-1]["panoptic_seg"] = panoptic_r
    return processed_results

def combine_uniform_patch_info(combine_cat_id, map_cat_id,  panoptic_seg, segments_info, class_id=9):
    for id_info in segments_info:
        print(id_info)
        if id_info['category_id']==class_id:
            print('found flower: cat id {} in cropped segment'.format(id_info['category_id']))
            if id_info['category_id'] not in combine_cat_id and id_info['isthing']==True:
                map_cat_id[id_info['category_id']] = id_info['id']
                combine_cat_id[id_info['category_id']] = id_info
            else:
                #area should be added since we maintain same cat_id and id
                #combine_cat_id[id_info['category_id']]['id'] = map_cat_id[id_info['category_id']]
                #change patch mask accordingly: same cat_id will maintain same instance id (mask label)
                if id_info['isthing']==True:
                    panoptic_seg[panoptic_seg==id_info['id']] = map_cat_id[id_info['category_id']]
                    if 'area' in id_info:
                        combine_cat_id[id_info['category_id']]['area'] += id_info['area']
            #else:
                #handle 'isthing'=='True'
        else:
            panoptic_seg[panoptic_seg == id_info['id']] = 0
    return combine_cat_id, map_cat_id,  panoptic_seg

def gen_unique_color(colorset):
    unique=False
    while not unique:
            color = tuple(np.random.choice(range(1, 255), size=3))
            if not(color in colorset):
                unique=True
                #colorset.append(color)
    return color

def rgb_mask_gen(colorsdict, img, segments_info, flower_metadata,  area_thr=0):

    segIDs = []
    bboxes = []
    areas = []
    h, w = img.shape
    #TODO: need instance predictions
    pred = _PanopticPrediction(img, segments_info, flower_metadata)
    # predicted instances
    all_instances = list(pred.instance_masks())
    # draw mask for all semantic segments first i.e. "stuff"
    for sem_mask, sinfo in pred.semantic_masks():
        category_idx = sinfo["category_id"]
        # 2: background class
        color = gen_unique_color(colorsdict)
        colorsdict.add(color)
        rgbmask = PIL.Image.new(mode="RGB", size=(w, h))
        rgbmask  = np.array(rgbmask)
        rgbmask[:, :, 0][sem_mask>0] = color[0]
        rgbmask[:, :, 1][sem_mask>0] = color[1]
        rgbmask[:, :, 2][sem_mask>0] = color[2]

        img_area = w * h
        flower_area = (sem_mask==0).sum() # 0:flower in semantic prediction
        img_area = img_area - flower_area

        segIDs.append(int(color[0] + 256 * color[1] + 256 * 256 * color[2]))
        areas.append(img_area)
        bboxes.append([0, 0, w, h])


    if len(all_instances)>0:
        masks, sinfo = list(zip(*all_instances))

        for i in range(len(all_instances)):
            category_ids = [x["category_id"] for x in sinfo]
            color = gen_unique_color(colorsdict)
            colorsdict.add(color)
            assert  masks[i].shape[0] == h and masks[i].shape[1]==w
            mask = GenericMask(masks[i], h, w)
            x0, y0, x1, y1 = mask.bbox()
            area_n = mask.area()

            rgbmask[:, :, 0][masks[i] != 0] = color[0]
            rgbmask[:, :, 1][masks[i] != 0] = color[1]
            rgbmask[:, :, 2][masks[i] != 0] = color[2]

            segID_n = int(color[0] + 256 * color[1] + 256 * 256 * color[2])
            x_n, y_n, w_n, h_n = x0, y0, (x1-x0)/2, (y1-y0)/2
            segIDs.append(segID_n)
            areas.append(area_n)
            bboxes.append([x_n, y_n, w_n, h_n])

    rgbmask = Image.fromarray(rgbmask.astype('uint8'), 'RGB')
    #assert flower_area==sum(areas[1:]), 'found {}=={}'.format(flower_area, sum(areas[1:]))
    return colorsdict, rgbmask, sem_mask, segIDs, bboxes, areas

def save_augmented_masks(final_mask, angle_set, outpath, im_name):
    for i, mask in enumerate(final_mask):
        mask[mask>0] = 255
        cv2.imwrite(os.path.join(outpath, 'masks_frame/{}_{}'.format(angle_set[i], os.path.basename(im_name))),
                    mask.cpu().numpy())


# %%function for generation [segment_info] portion of the panoptic annotation
def pan_ann_gen(segIDs, imgID, bboxes, areas, img_area=None):  # RLEs is not a necessary argument
    segment_info = []
    for i in range(len(segIDs)):
        area = areas[i]
        bbox = bboxes[i]
        segID = segIDs[i]
        if i == 0:
            print('found panoptic background category of area {}'.format(area))
            catID = 0
            new_seg_pan = {
                'id': segID,
                'category_id': catID,
                'area': area,
                'bbox': bbox,
                'iscrowd': 0
            }
            segment_info.append(new_seg_pan)
        else:
            catID = 1
            new_seg_pan = {
                'id': segID,
                'category_id': catID,
                'area': area,
                'bbox': bbox,
                'iscrowd': 0
            }
            segment_info.append(new_seg_pan)

    filename = str(int(imgID)) + '.png'
    pan_ann = {
        'segments_info': segment_info,
        'image_id': imgID,
        'file_name': filename
    }

    return pan_ann

def save2panopticJSON(imgID, panoptic_seg, segments_info, flower_metadata,
                      imgs, pan_anns, inst_anns, colorsdict, rgbmask_dir, sem_mask_dir):

    img = panoptic_seg.to("cpu")
    h, w = img.shape
    img_area = w * h
    colorsdict, rgbmask, sem_seg, segIDs, bboxes, areas = rgb_mask_gen(colorsdict, img, segments_info,
                                                              flower_metadata, area_thr=contour_area_thr)
    #save panoptic rgb mask
    rgbmask.save(rgbmask_dir + '{:05d}.png'.format(int(imgID)))
    #save sem mask
    semmask = PIL.Image.new(mode="RGB", size=(w, h))
    semmask = semmask.convert('L')
    semmask = np.array(semmask)
    semmask[sem_seg == 0] = 0
    semmask[sem_seg != 0] = 255
    Image.fromarray(semmask).save(sem_mask_dir + '{:05d}.png'.format(int(imgID)))


    flower_area = sum(areas[1:])
    img_area = img_area - flower_area
    pan_ann = pan_ann_gen(segIDs, imgID, bboxes, areas,
                        img_area=img_area)
    pan_anns.append(pan_ann)

    # inst_ann=inst_ann_gen(outers, segIDs, imgID, bboxes, areas)
    # inst_anns.extend(inst_ann)

    img_n = {
        'license': None,
        'file_name': '{:05d}.png'.format(int(imgID)),
        'coco_url': None,
        'height': h,  # height
        'width': w,  # width
        'date_captured': None,
        'flickr_url': None,
        'id': imgID
    }
    imgs.append(img_n)

    return imgs, pan_anns, inst_anns, colorsdict

def save_augmented_image(imgrot, flower_metadata, panoptic_seg,
                         segments_info, outpath, im_name, angle, class_id):
    #imgrot = np.zeros(imgrot.shape, dtype='uint8')
    v = Visualizer(imgrot[:, :, ::-1],
                   metadata=flower_metadata,
                   scale=0.5,)
                   #instance_mode=ColorMode.IMAGE_BW)
    ##remove the colors of unsegmented pixels. This option is only available for segmentation models
    if len(class_id)==2:
        segments_info = segments_info
    else:
        segments_info = [info for info in segments_info if info['category_id'] == class_id]
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info, alpha=0.15)
    cv2.imwrite(os.path.join(outpath, 'aug_images/{}_{}'.format(angle, os.path.basename(im_name))),
                out.get_image()[:, :, ::-1])

def delete_all(demo_path, fmt='png'):
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)

def get_test_config(pretrained=False, test_aug=False, dataset='AppleA'):
    if test_aug:
        angleSet = [0, 6, 12, 18, 24, 72, 78, 84, 90, 96, 102, 162,
                    168, 174, 180, 186, 192, 252, 258, 264, 270, 276,
                    272,336, 342, 348, 354]
    else:
        angleSet = [0]
    if pretrained:
        if test_aug:
            class_id = 9
        else:
            class_id = [9]
    else:
        if test_aug:
            class_id = 0
        else:
            class_id = [0, 1]

    #default config
    cfg = get_cfg()

    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    if pretrained:
        flower_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    else:
        register_coco_panoptic_separated(name=dataset+"_test",
                                         metadata={},
                                         image_root=img_dir,
                                         panoptic_root=panoptic_masks,
                                         panoptic_json=panoptic_json,
                                         sem_seg_root=sem_seg_masks,
                                         instances_json=instance_json)
        for d in ['train']:
            #DatasetCatalog.register("AppleA_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
            MetadataCatalog.get(dataset+'_' + d).set(thing_classes=["flower"],
                                                   stuff_classes=["flower", "background"])
        flower_metadata = MetadataCatalog.get(dataset+"_test")


        cfg.MODEL.DEVICE = 1
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
        cfg.DATASETS.TEST = (dataset+"_split",)
        cfg.MODEL.BACKBONE.FREEZE_AT = 4
        # Inference with a panoptic segmentation model
        if pretrained:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        else:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
            cfg.MODEL.PANOPTIC_FPN.TUFF_AREA_LIMIT = 50#4096 #default:4096
            cfg.MODEL.PANOPTIC_FPN.INSTANCES_CONFIDENCE_THRESH = 0.5 #default: 0.5
            cfg.MODEL.PANOPTIC_FPN.OVERLAP_THRESH = 0.5 #default: 0.5 ??
            cfg.MODEL.WEIGHTS = '/home/abubakarsiddique/trainPanopticModel/flower_model/model_final.pth'

        #obj_detect = build_model(cfg)
        #obj_detect.eval()
        #obj_detect.cuda()  # device=self.gpu
        #checkpointer = DetectionCheckpointer(obj_detect)
        #checkpointer.load(cfg.MODEL.WEIGHTS)
    return cfg, angleSet, class_id, flower_metadata

##newer version of this method needed???
def define_categories_dictionary():
    categories = [
        {
            'id': 1,
            'name': 'flower',
            'supercategory': 'flower',
            'isthing': 1,
            'color': []
        },
        {
            'id': 0,
            'name': 'background',
            'supercategory': 'background',
            'isthing': 0,
            'color': []
        }]
    return categories

#TODO: implement panoptic format of test results for evaluation
if __name__ == '__main__':

    dataset = 'Pear'
    det2 = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/Detectron2'
    NAS = '/media/NAS/LabFiles/Walden'
    if dataset!='AppleA':
        base_dir = NAS + '/otherFlowerDatasets/'+dataset+'/dataFormattedProperly/splitImages4x3'
    else:
        base_dir=NAS + '/trainTestSplit/test/dataFormattedProperly/splitImages4x3'
    img_dir = base_dir + '/flowersSplit'
    panoptic_masks = base_dir + '/rgbMasksSplit'
    panoptic_json = base_dir + '/'+dataset+'_panoptic_split4x3_test.json'

    # 0: flower, 255: background
    sem_seg_masks = base_dir + '/semMasksSplit'
    instance_json = base_dir + '/'+dataset+'_instance_split4x3_test.json'


    pretrained = 0

    regress_cluster= 0
    test_aug = 0
    save_aug = 0
    #data = 'AppleA'
    save_infer_panoptic_json = 1
    regress_model_type = 'instance_segmentation'

    #for eval results
    #-----------------------------------------------------------------------
    outpath='/home/abubakarsiddique/trainPanopticModel/all_test_results/'+dataset+'_test/'
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

    #global variable to keep pan anoos for every frame
    pan_anns = []
    inst_anns = []
    imgs = []
    colorsdict = set()
    #-----------------------------------------------------------------------
    #prepare configuration file
    cfg, angleSet, class_id, flower_metadata = get_test_config(pretrained=pretrained, test_aug=test_aug,dataset=dataset)
    #configure test model
    predictor = DefaultPredictor(cfg)

    #cv2.namedWindow('image')
    #cv2.resizeWindow('image', 1920, 1080)

    #images = sorted(glob.glob(storage + 'tracking_wo_bnw/data/CLASP1/train_gt/' + cam + '/img1/*.png'))
    #images = sorted(glob.glob(storage + 'Detectron2/flower_dataset/AppleA_renamed/FlowerImages/*.jpg'))
    images = sorted(glob.glob(base_dir + '/flowersSplit/*.jpg'))
    for im_name in images:
        fr_num = float(os.path.basename(im_name).split('.')[0])
        if fr_num>0:
            print('Data {} Frame {}'.format(dataset, os.path.basename(im_name)))
            im = cv2.imread(im_name)
            #im = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2))
            patch_scale = 1
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

                    if test_aug:
                        panoptic_seg, segments_info = predictor(imgrot)["panoptic_seg"]
                        if save_aug:
                            save_augmented_image(imgrot, flower_metadata, panoptic_seg,
                                                 segments_info, outpath, im_name, angle, class_id)
                        #unifor patch infos
                        combine_cat_id, map_cat_id, panoptic_seg = combine_uniform_patch_info(combine_cat_id,
                                                                                              map_cat_id,
                                                                                              panoptic_seg,
                                                                                              segments_info,
                                                                                              class_id=class_id)
                        # remap seg for flower class
                        panoptic_seg = mask_image_remap(panoptic_seg, im, angle, even_scale=patch_scale)
                        segms.append(panoptic_seg)
                    else:
                        panoptic_seg, segments_info = predictor(imgrot)["panoptic_seg"]


                if test_aug:
                    panoptic_seg = merge_segms(segms, segms_info, even_scale=patch_scale, im_mask=img_mask)
                    img_segms_info = list(combine_cat_id.values())
                    if class_id in map_cat_id:
                        panoptic_seg[panoptic_seg==map_cat_id[class_id]] = 1
                        panoptic_seg[panoptic_seg != 1] = 0
                        final_mask.append(panoptic_seg)

                        #rgb_img = cv2.cvtColor(bmask, cv2.COLOR_GRAY2RGB)

                        #cv2.imwrite(os.path.join(outpath, 'mask/{}_{}'.format(angle, os.path.basename(im_name))),
                                   #rgb_img)
                        #v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
                        #out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), img_segms_info)
                        #cv2_imshow(out.get_image()[:, :, ::-1])
                        #cv2.imwrite(os.path.join(outpath, '{}_{}'.format(angle, os.path.basename(im_name))),
                                    #out.get_image()[:, :, ::-1])
                else:

                    save_augmented_image(imgrot, flower_metadata, panoptic_seg,
                                         segments_info, outpath, im_name, angle, class_id)

                    imgs, pan_anns, \
                    inst_anns, colorsdict = save2panopticJSON(int(fr_num),  panoptic_seg,
                                                              segments_info, flower_metadata,
                                                              imgs, pan_anns, inst_anns, colorsdict,
                                                              rgbmask_dir, sem_mask_dir)

                #remap bbox and mask with vis angle==0, collect also the rotated boxes

                #accumulate all the augmented dets

            if regress_cluster:
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
            if test_aug: #class_id in map_cat_id and
                final_masks = torch.cat(([mask.unsqueeze(0).type(torch.float) for mask in final_mask]), dim=0)
                final_masks = final_masks.mean(0)
                final_masks[final_masks >= 0.5] = 255
                final_masks[final_masks < 0.5] = 0
                # during SSL iteration split the rgb image and
                # corresponding mask when flower detection available

                cv2.imwrite(os.path.join(outpath, 'mask/{}'.format(os.path.basename(im_name))),
                            final_masks.cpu().numpy())
                assert len(img_segms_info) == 1 and img_segms_info[0]['id'] == map_cat_id[class_id]
                rgb_mask[final_masks > 0] = map_cat_id[class_id] #img_segms_info[0]['id']  # map_cat_id[9]
                v = Visualizer(im[:, :, ::-1],
                               metadata=flower_metadata,
                               scale=0.5,)
                               #instance_mode=ColorMode.IMAGE_BW)
                out = v.draw_panoptic_seg_predictions(rgb_mask.to("cpu"), img_segms_info)
                # cv2_imshow(out.get_image()[:, :, ::-1])
                cv2.imwrite(os.path.join(outpath, '{}'.format(os.path.basename(im_name))),
                        out.get_image()[:, :, ::-1])
        #prepare and update training data dictionary

    #call training subroutine
    #cv2.destroyAllWindows()
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


# IoU metric
p_masks = sorted(glob.glob(sem_mask_dir +'*.png'))
g_masks = sorted(glob.glob(base_dir+'/bMasksSplit/*.png'))
#p_masks=sorted(os.listdir(sem_mask_dir))
#g_masks=sorted(os.listdir(sem_seg_masks))
ious = []
F1s = []
Prs = []
Rclls = []

pred=[]
gt=[]

for y_pred, y_true in zip(p_masks, g_masks):

    gt_mask = cv2.imread(y_true, 0)
    #if 255 in gt_mask:
    gt_mask[gt_mask>0] = 1

    pred_mask = cv2.imread(y_pred,0)
    pred_mask[pred_mask == 0] = 1
    pred_mask[pred_mask == 255] = 0

    pred.extend(pred_mask.flatten())
    gt.extend(gt_mask.flatten())
    # jscore = jaccard_score(gt_mask.flatten(), pred_mask.flatten(), average='binary', pos_label=1)
    # F1 = f1_score(gt_mask.flatten(), pred_mask.flatten(), average='binary', pos_label=1)
    # Pr = precision_score(gt_mask.flatten(), pred_mask.flatten(), average='binary', pos_label=1)
    # Rcll = recall_score(gt_mask.flatten(), pred_mask.flatten(), average='binary', pos_label=1)

    # print('jscore: {} between gt: {}, pred: {}'.format(jscore, float(os.path.basename(y_true).split('.')[0]), float(os.path.basename(y_pred).split('.')[0])))
    # ious.append(jscore)
    # F1s.append(F1)
    # Prs.append(Pr)
    # Rclls.append(Rcll)

jscore = jaccard_score(gt, pred, average='binary', pos_label=1)
F1 = f1_score(gt, pred, average='binary', pos_label=1)
Pr = precision_score(gt, pred, average='binary', pos_label=1)
Rcll = recall_score(gt, pred, average='binary', pos_label=1)
print(dataset)
print('IoU: ' + str(jscore))
print('F1: ' + str(F1))
print('Pr: ' + str(Pr))
print('Rcll: ' + str(Rcll))

# print(dataset)
# print('IoU: {}'.format(np.nanmean(ious)))
# print('F1: {}'.format(np.nanmean(F1s)))
# print('Pr: {}'.format(np.nanmean(Prs)))
# print('Rcll: {}'.format(np.nanmean(Rclls)))