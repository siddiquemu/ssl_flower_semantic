import copy
import copyreg

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
from collections import OrderedDict
import collections

#import imgaug.augmenters as iaa
#import imgaug as ia

# In[2]:
'''
{
    'id': 2,
    #'id': 31,   #val 2017
    'name': 'bag',
    'supercategory': 'accessory',
},
{
   #'id': 2,
    'id': 27,   #val 2017
    'name': 'backpack',
    'supercategory': 'accessory',
},
{
#    'id': 2,
    'id': 33,   #val 2017
    'name': 'suitcase',
    'supercategory': 'accessory',
},
'''
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

def define_dataset_dictionary_pan(mode='instance'):
    """Training data structures for the panoptic model
    """
    dataset = {
        'info': {},
        'images': [],
        'licenses': [],
        'annotations': [],
        'categories': [],
    }
    if mode=='panoptic':
        dataset['categories']=[
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
            'isthing':0,
            'color':[]
            }]
    elif mode=='instance':
        dataset['categories']=[
            {
            'id': 1,
            'name': 'flower',
            'supercategory': 'flower',
            'isthing': 1,
            'color': []
            }]
    return dataset


def Write_Segments_info_pan(segment_infos, bbox, id, catId, area, instance_certainty=-1):
    """Update the segments info iteratively for each instance
    :param bbox: 4D top-left and width height
    :param id: segment id = int(color[0] + 256 * color[1] + 256 * 256 * color[2])
    :param img_id: index of image
    :param catId: category_id> 0: background, 1: flower
    :param area: area of the segment
    :param dataset: global dictionary object for the panoptic json
    :param instance_certainty: default: -1 to indicate as unused, otherwise: cluster score in test_aug
    :return: dataset: global dictionary object for the panoptic json
    """
    info = {"area": area,
            "iscrowd": 0,
            "bbox": bbox,
            "category_id": catId,
            "id": id,
            "instance_certainty": instance_certainty
            }

    new_dict = collections.OrderedDict()
    new_dict["area"] = info["area"]
    new_dict["iscrowd"] = info["iscrowd"]
    new_dict["bbox"] = info["bbox"]
    new_dict["category_id"] = info["category_id"]
    new_dict["id"] = info["id"]
    new_dict["instance_certainty"] = info["instance_certainty"]

    segment_infos.append(new_dict)

    return segment_infos

def init_annotations_info_pan(dataset, segment_infos, imgID, filename):
    """Initiate segments info dict once for eaach frame
    """
    info = {
    'segments_info': segment_infos,
    'image_id': imgID,
    'file_name': filename
    }
    new_dict = collections.OrderedDict()
    new_dict['segments_info'] = info['segments_info']
    new_dict["image_id"] = info["image_id"]
    new_dict['file_name'] = info['file_name']

    dataset['annotations'].append(new_dict)

    return dataset


def define_dataset_dictionary ():
    dataset = {
        'info': {},
        'images': [],
        'licenses': [],
        'annotations': [],
        'categories': [],
    }

    dataset['categories'] = [
        {
            'id': 1,
            'name': 'person',
            'supercategory': 'person',
        },
        {
            'id': 2,
            'name': 'bag',
            'supercategory': 'bag',
        },
    ]
    return dataset

def define_dataset_dictionary_mp (manager, database='flower', mode='panoptic'):
    dataset = manager.dict()
    dataset['info'] = manager.dict()
    dataset['images'] = manager.list()
    dataset['licenses'] = manager.list()
    dataset['annotations'] = manager.list()

    if database=='flower':
        if mode=='panoptic':
            dataset['categories']=[
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
                'isthing':0,
                'color':[]
                }]
        elif mode=='instance':
            dataset['categories']=[
                {
                'id': 1,
                'name': 'flower',
                'supercategory': 'flower',
                'isthing': 1,
                'color': []
                }]
    elif database in ['clasp1', 'clasp2']:
        dataset['categories'] = [
            {
                'id': 1,
                'name': 'person',
                'supercategory': 'person',
            },
            {
                'id': 2,
                'name': 'bag',
                'supercategory': 'bag',
            },
        ]
    else:
        print(f'{database} should be clasp1, clasp2, or flower')
    return dataset

def Proxy2RealObj(proxy_objs, database='clasp1'):
    # proxy_objs: should be DictProxy obj
    #The purpose of the DictProxy class is to give you a dict that is safe to share across
    # processes, which means that
    # it must implement some locking on top of the normal dict functions.
    if database in ['clasp1', 'clasp2']:
        dataset_clasp = define_dataset_dictionary ()
    elif database=='flower':
        dataset_clasp = define_flower_dataset_dictionary ()
    else:
        print(f'{database} should be clasp1 or clasp2 or flower')
    #case1: list or dict items in Proxy object
    #if isinstance(proxy_objs, object) and not isinstance(proxy_objs, (dict, list)):
    if proxy_objs.__repr__().split()[0]== '<DictProxy':
        dict_objs = dict(proxy_objs)
        #{'info': <DictProxy object, typeid 'dict' at 0x7fdbd20896a0>,
        # 'images': <ListProxy object, typeid 'list' at 0x7fdbd2089940>,
        # 'licenses': <ListProxy object, typeid 'list' at 0x7fdbd2089320>,
        # 'annotations': <ListProxy object, typeid 'list' at 0x7fdbd20895f8>,
        # 'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'}, {'id': 2, 'name': 'bag', 'supercategory': 'bag'}]}
    for key, value in dict_objs.items():
        if dict_objs[key].__repr__().split()[0] == '<ListProxy':
            dataset_clasp[key] = list(value)
        elif dict_objs[key].__repr__().split()[0] == '<DictProxy':
            dataset_clasp[key] = dict(value)
        else:
            print('only allow ListProxy and DictProxy')
    return dataset_clasp

def define_flower_dataset_dictionary ():
    dataset = {
        'info': {},
        'images': [],
        'licenses': [],
        'annotations': [],
        'categories': [],
    }

    dataset['categories'] = [
        {
            'id': 1,
            'name': 'flower',
            'supercategory': 'flower',
        },
    ]
    return dataset

def Write_To_Json(filename, data):
    #file = './' + path + './' + filename + '.json'
    file = filename
    with open(file,'w') as fp:
        json.dump(data, fp,cls=NpEncoder)

def Write_ImagesInfo(img, filename, id, dataset):
    info = {"id": id,
            "file_name": filename,
            "width": img.shape[1],
            "height": img.shape[0],
            "date_captured": "",
            "license": "",
            "coco_url": "",
            "flickr_url": ""}
    new_dict = collections.OrderedDict()
    new_dict["license"] = info["license"]
    new_dict["file_name"] = info["file_name"]
    new_dict["coco_url"] = info["coco_url"]
    new_dict["height"] = info["height"]
    new_dict["width"] = info["width"]
    new_dict["date_captured"] = info["date_captured"]
    new_dict["flickr_url"] = info["flickr_url"]
    new_dict["id"] = info["id"]
    if new_dict not in dataset['images']:
        #print('duplicate image', id)
        dataset['images'].append(new_dict)
    return dataset

def Write_AnnotationInfo(bbox, segmentation, img_id, id, catId, area, dataset, instance_certainty=-1):
    """
    :param bbox: 4D top-left and width height
    :param segmentation: 2D point set of target boundary
    :param img_id: index of image
    :param id: annotation id
    :param catId:
    :param area:
    :param dataset:
    :param instance_certainty: default: -1 to indicate as unused, otherwise: cluster score in test_aug
    :return:
    """
    info = {"segmentation": segmentation,
           "area": area,
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": bbox,
            "category_id": catId,
            "id": id,
            "instance_certainty": instance_certainty
            }

    new_dict = collections.OrderedDict()
    new_dict["segmentation"] = info["segmentation"]
    new_dict["area"] = info["area"]
    new_dict["iscrowd"] = info["iscrowd"]
    new_dict["image_id"] = info["image_id"]
    new_dict["bbox"] = info["bbox"]
    new_dict["category_id"] = info["category_id"]
    new_dict["id"] = info["id"]
    new_dict["instance_certainty"] = info["instance_certainty"]
    dataset['annotations'].append(new_dict)
    return dataset

def load_flower_json(json_path, catID=1):
    coco_flower = COCO(json_path)
    imgIds = coco_flower.getImgIds(catIds=catID )
    print ('total images of flower is', len(imgIds))
    print ('Their images ids are', imgIds)
    return coco_flower

def load_clasp_json(json_path, percent=1):
    coco_clasp = COCO(json_path)
    #imgIds = set(list(coco_clasp.getImgIds(catIds=1)))
    #frames = list(imgIds.union(set(list(coco_clasp.getImgIds(catIds=2)))))
    imgIds = coco_clasp.getImgIds()
    print ('total images {} of {} percent annotations'.format(len(imgIds), percent))
    print ('Their images ids are', imgIds)
    return coco_clasp, imgIds


def get_frame_anns(coco_clasp, im_id):
    # return anns for an image
    Cur_imgId = coco_clasp.getImgIds(imgIds=[im_id])
    img = coco_clasp.loadImgs(Cur_imgId)[0]
    annIds = coco_clasp.getAnnIds(imgIds=img['id'])
    #annIds.extend(coco_clasp.getAnnIds(imgIds=img['id'], catIds=2, iscrowd=0))
    allanns = coco_clasp.loadAnns(annIds)
    return allanns


if __name__ == '__main__':
    #change the category to your need
    #catIds = coco.getCatIds(catNms=['suitcase'])
    # coco = COCO('/home/chumingwen/Desktop/coco_images/train/Annotation/Train_Annotation.json')
    coco = COCO('/home/chumingwen/Desktop/coco_2017/annotations/instances_train2017.json')
    catIds=[33]
    imgIds_suitcase = coco.getImgIds(catIds=catIds )
    print ('total images of suitcase is', len(imgIds_suitcase))
    print ('Their images ids are', imgIds_suitcase)


    #catIds = coco.getCatIds(catNms=['backpack'])
    catIds=[27]
    imgIds_backpack = coco.getImgIds(catIds=catIds)
    print ('total images of backpack is', len(imgIds_backpack))
    print ('Their images ids are', imgIds_backpack)


    #catIds = coco.getCatIds(catNms=['handbag'])
    catIds=[31]
    imgIds_handbag = coco.getImgIds(catIds=catIds)
    print ('total images of handbag categoty is', len(imgIds_handbag))
    print ('Their images ids are', imgIds_handbag)




    allIds = [imgIds_suitcase, imgIds_backpack, imgIds_handbag]
    annotationcnt = 0


    for i in range(3):
        print ('start')
        num=1

        imgIds = allIds[i]
        if i==0:
            #suitcase
            catIds = [33]
        #    newcatId = [4]
        elif i==1:
            #backpack
            catIds = [27]
        #    newcatId = [3]
        else:
            #handbag
            catIds = [31]
        #    newcatId = [2]
        newcatId = [2]
        for id in imgIds:
            #id = 125952
            anglecount = 1
            Cur_imgId = coco.getImgIds(imgIds = [id])
            img = coco.loadImgs(Cur_imgId)[0]
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=0)


            imgfilename = img['file_name']
            filename = '/home/chumingwen/Desktop/coco_2017/train2017/' + imgfilename

            img = cv2.imread(filename)

            listpolys = []



            allanns = coco.loadAnns(annIds)


            for anns in allanns:
                b = anns['bbox']
                annotationcnt += 1

                poly1 = ia.Polygon([])
                polys = [poly1]


                for seg in anns['segmentation']:
                    newpts = np.asarray(seg).reshape(-1, 2)
                    poly = ia.Polygon(newpts)
                    polys.append(poly)
                    newpolys = ia.PolygonsOnImage(polys, shape=img.shape)

                listpolys.append(newpolys)



            # rotate each image 8 times
            for rotdegree in range(0, 360, 45):

                imgname = imgfilename.split('.')[0] + str(anglecount) + '.jpg'
                imgnamemask = str(newcatId[0]) + imgfilename.split('.')[0] + str(anglecount) + '.jpg'
                imgIdnew = 100 * Cur_imgId[0] + anglecount


                writepath = '/home/chumingwen/Desktop/coco_2017/train2017_aug/' +  imgname
                writepathmask = '/home/chumingwen/Desktop/coco_2017/train2017_augmask/' + imgnamemask


                # rotate image
                aug = iaa.Affine(rotate=rotdegree, fit_output=True)
                imgrot = aug.augment_image(img)

                # save new image and save image info to dataset
                cv2.imwrite(writepath, imgrot)
                Write_ImagesInfo(imgrot, imgname, imgIdnew)


                # rotate polygon mask and create bounding box
                imgmask = imgrot
                cnt = 0
                p = False
                for newpolys in listpolys:
                    rotpolys = aug.augment_polygons(newpolys)
                    area = 0
                    segrot = []

                    bbox = rotpolys.polygons[1].to_bounding_box()


                    for polyrot in rotpolys.polygons[1:]:

                        area += polyrot.area

                        extpts = polyrot.exterior.reshape(1, -1).tolist()
                        segpartrot = [round(x, 2) for x in extpts[0]]

                        for i in range(len(segpartrot)):
                            segpartrot[i] = max(0, segpartrot[i])
                            if i%2 == 0:
                                segpartrot[i] = min(imgrot.shape[1], segpartrot[i])
                            else:
                                segpartrot[i] = min(imgrot.shape[0], segpartrot[i])


                        segrot.append(segpartrot)

                        seg1 = segpartrot
                        newpts1 = np.asarray(seg1).reshape(-1, 2)
                        poly2 = ia.Polygon(newpts1)
                        polys1 = [poly2, poly1]
                        newpolys1 = ia.PolygonsOnImage(polys1, shape=imgrot.shape)
                        polyrot1 = newpolys1.polygons[0]
                        imgmask = polyrot1.draw_on_image(imgmask, color=(255, 0, 255),alpha=0.7, size=1)


                        bboxnew = polyrot.to_bounding_box()
                        bbox = bbox.union(bboxnew)

                    x1 = max(0, bbox.x1)
                    y1 = max(0, bbox.y1)
                    x2 = min(imgrot.shape[1], bbox.x2)
                    y2 = min(imgrot.shape[0], bbox.y2)




                    bboxrot = [x1, y1, x2 - x1, y2 - y1]
                    if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0 or bboxrot[2]<0 or bboxrot[3]<0 :
                        print('neg error bbox')
                    bboxfinal = [round(x, 2) for x in bboxrot]



                    cv2.rectangle(imgmask, (int(bboxfinal[0]), int(bboxfinal[1])), (int(bboxfinal[0]+bboxfinal[2]), int(bboxfinal[1] + bboxfinal[3])), (0, 255, 0), 1)

                    #plt.imshow(imgmask)
                    #plt.show()


                    #print(bboxfinal)
                    #print(segrot)


                # write annotation to dataset
                    annIdnew = 100 * annIds[cnt] + anglecount
                    if area<10:
                        print('small area', area)
                    if bboxfinal[2]<3 or bboxfinal[3]<3:
                        print('small w or h', bboxfinal[2], bboxfinal[3], annIdnew, imgIdnew)
                        #plt.imshow(imgmask)
                        #plt.show()
                    Write_AnnotationInfo(bboxfinal, segrot, imgIdnew, annIdnew, newcatId[0], area)
                    cnt += 1

                    cv2.imwrite(writepathmask, imgmask)

                anglecount += 1


            print (num)
            num+=1

    print ('bags annotation count: ', annotationcnt)
    print ('bags done')


    # person
    personann = 0
    catIds = [1]
    imgIds_person = coco.getImgIds(catIds=catIds )
    print ('total images of person is', len(imgIds_person))
    print ('Their images ids are', imgIds_person)


    for id in imgIds_person:
        Cur_imgId = coco.getImgIds(imgIds=[id])
        img = coco.loadImgs(Cur_imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=0)
        allanns = coco.loadAnns(annIds)
        imgfilename = img['file_name']
        filename = '/home/chumingwen/Desktop/coco_2017/train2017/' + imgfilename
        #filename = '/home/chumingwen/Desktop/coco_images/train/All_Images/' + imgfilename
        img = cv2.imread(filename)

        writepath = '/home/chumingwen/Desktop/coco_2017/train2017_aug/' + imgfilename
        cv2.imwrite(writepath, img)


        Write_ImagesInfo(img, imgfilename, id)

        for anns in allanns:
            personann +=1
            Write_AnnotationInfo(anns['bbox'], anns['segmentation'], anns['image_id'], anns['id'], catIds[0], anns['area'])

    print ('person annotation count: ', personann)


    #path = './'
    #savefilename = '/home/chumingwen/Desktop/coco_2017/coco_2017_aug_train'
    #Write_To_Json(savefilename, dataset)