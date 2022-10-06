import os
from utils import delete_all

def get_dirs(dataset = 'Peach_test'):
    dataset=dataset
    NAS = '/media/NAS/LabFiles/Walden'
    det2 = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/Detectron2'

    if dataset == 'AppleA_train':
        train_dir = det2 + '/flower_dataset/AppleA_Panoptic/DataFormattedProperly/SplitImages4x3/train/'

        NAS = '/media/NAS/LabFiles/Walden'
        base_dir = NAS + '/trainTestSplit/train/dataFormattedProperly/splitImages4x3'

        img_dir = base_dir + '/flowersSplit'
        panoptic_masks = base_dir + '/rgbMasksSplit'
        panoptic_json = base_dir + '/appleA_panoptic_split4x3_train.json'

        # 0: flower, 255: background
        sem_seg_masks = base_dir + '/semMasksSplit'
        instance_json = base_dir + '/appleA_instance_split4x3_train.json'

    if dataset == 'AppleA_test':
        data = 'AppleA'
        base_dir = NAS + '/trainTestSplit/test/dataFormattedProperly/splitImages4x3'

        img_dir = base_dir + '/flowersSplit'
        panoptic_masks = base_dir + '/rgbMasksSplit'
        panoptic_json = base_dir + '/appleA_panoptic_split4x3_test.json'

        # 0: flower, 255: background
        sem_seg_masks = base_dir + '/semMasksSplit'
        instance_json = base_dir + '/appleA_instance_split4x3_test.json'
        img_fmt = '.jpg'

    if dataset == 'AppleB_test':
        data = 'AppleB'
        base_dir = NAS + '/otherFlowerDatasets/AppleB/dataFormattedProperly/splitImages4x3'

        img_dir = base_dir + '/flowersSplit'
        panoptic_masks = base_dir + '/rgbMasksSplit'
        panoptic_json = base_dir + '/AppleB_panoptic_split4x3_test.json'

        # 0: flower, 255: background
        sem_seg_masks = base_dir + '/semMasksSplit'
        instance_json = base_dir + '/AppleB_instance_split4x3_test.json'

        img_fmt = '.bmp'

    if dataset == 'Peach_test':
        data = 'Peach'
        base_dir = NAS + '/otherFlowerDatasets/Peach/dataFormattedProperly/splitImages4x3'

        img_dir = base_dir + '/flowersSplit'
        panoptic_masks = base_dir + '/rgbMasksSplit'
        panoptic_json = base_dir + '/Peach_panoptic_split4x3_test.json'

        # 0: flower, 255: background
        sem_seg_masks = base_dir + '/semMasksSplit'
        instance_json = base_dir + '/Peach_instance_split4x3_test.json'

        img_fmt = '.bmp'

    if dataset == 'Pear_test':
        data = 'Pear'
        base_dir = NAS + '/otherFlowerDatasets/Pear/dataFormattedProperly/splitImages4x3'

        img_dir = base_dir + '/flowersSplit'
        panoptic_masks = base_dir + '/rgbMasksSplit'
        panoptic_json = base_dir + '/Pear_panoptic_split4x3_test.json'

        # 0: flower, 255: background
        sem_seg_masks = base_dir + '/semMasksSplit'
        instance_json = base_dir + '/Pear_instance_split4x3_test.json'

        img_fmt = '.bmp'

    # for eval results
    # -----------------------------------------------------------------------
    outpath = os.path.join('/home/abubakarsiddique/trainPanopticModel', dataset)
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
    return data, img_dir, panoptic_masks, sem_seg_masks, \
           panoptic_json, instance_json, outpath, \
           rgbmask_dir, sem_mask_dir, img_fmt, base_dir