from clasp2coco import load_clasp_json, load_flower_json
import os
import numpy as np
import pdb
import glob
import pandas as pd

def delete_all(demo_path, fmt='png'):
    try:
        filelist = glob.glob(os.path.join(demo_path, '*'))
        if len(filelist) > 0:
            for f in filelist:
                os.remove(f)
    except:
        print(f'{demo_path} is already empty')

def get_all_dirs(args, exp, init_params, storage, models_dir, model_type, percent_gt=None,):
    # datasets and model catalogs for Self-SL or Semi-SL

    if init_params['database']=='flower':
        # source images dirs
        data_root = f'{storage}/tracking_wo_bnw/data/{args.database}/train_gt_panoptic_sw_16_8'
        #init_params['train_img_dir'] = storage + f'/SoftTeacher/data/{args.database}/trainFlower/'
        #init_params['unlabeled_img_dir'] = storage + f'/SoftTeacher/data/{args.database}/unlabeledFlower/'
        # init_params['train_img_dir'] = storage + f'/tracking_wo_bnw/data/{args.database}/train_gt_sw/trainFlowerAug/'
        # init_params['unlabeled_img_dir'] = storage + f'/tracking_wo_bnw/data/{args.database}/train_gt_sw/unlabeledFlower/'
        # init_params['train_img_dir'] = storage + f'/tracking_wo_bnw/data/{args.database}/train_gt_sw/trainFlowerAug/'
        # init_params['unlabeled_img_dir'] = storage + f'/tracking_wo_bnw/data/{args.database}/train_gt_sw/unlabeledFlowerAppleA/'

        if args.data_set=='AppleA_train':
            init_params['train_img_dir'] = f'{data_root}/trainFlowerAug/'
            # if not os.path.exists(init_params['train_img_dir']):
            #     os.makedirs(init_params['train_img_dir'])
            # # else:
            # #     delete_all(init_params['train_img_dir'])
        else:           
            #init_params['unlabeled_img_dir'] = f'{storage}/tracking_wo_bnw/data/{args.database}/train_gt_panoptic_sw/unlabeledFlower{args.data_set}/'
            init_params['unlabeled_img_dir'] = f'{data_root}/unlabeledFlower{args.data_set}/'
            # init_params['panoptic_masks'] = f'{storage}/panoptic_labels'
            # init_params['panoptic_json'] = f'{storage}/panoptic_train_2021.json'

            # #semantic mask> 0: flower, 255: background
            # init_params['sem_seg_masks'] = f'{storage}/semantic_labels'
            # init_params['instance_json'] = f'{storage}/instances_train_2021.json'
        


        if model_type=='modified_loss_semi':
            #read tau_seg from test/val evaluation using previous model
            if not init_params['apply_rgr']:
                #tau_seg = pd.read_csv(f'{data_root}/SSL_Data/{args.data_set}/CV{args.CV}/tau_seg_iter{exp-1}.csv')
                init_params['remap_score_thr'] = 0.15#tau_seg.values[0][0]
                #print(f'found score threshold from maximum F1: {tau_seg.values[0][0]}')
            else:
                init_params['remap_score_thr'] = 0.15
            #training data dirs
            benchmark = storage + f'/tracking_wo_bnw/data/{args.database}/'
            init_params['output_dir'] = benchmark + f'test_aug_gt_pan/iter{exp}'
            
            if args.number_gpus>2: # use local storage in remote server
                init_params['output_dir'] = f"{init_params['local_storage']}/iter{exp}"
                
            if not os.path.exists(init_params['output_dir']):
                os.makedirs(init_params['output_dir'])

            # images and JSON file dirs for unlabeled data_set
            init_params['sem_seg_masks'] = init_params['output_dir'] + '/semantic_labels'
            init_params['instance_json'] = init_params['output_dir'] + f'/instances_{args.database}_test_aug_{exp}.json'

            init_params['panoptic_masks'] = init_params['output_dir'] + '/panoptic_labels'
            init_params['panoptic_json'] = init_params['output_dir'] + f'/panoptic_{args.database}_test_aug_{exp}.json'
            init_params['AugImgDir'] = init_params['output_dir'] + f'/img1_{exp}'
            
            if not os.path.exists(init_params['AugImgDir']):
                os.makedirs(init_params['AugImgDir'])
                os.makedirs(init_params['sem_seg_masks'])
                os.makedirs(init_params['panoptic_masks'])
            else:
                delete_all(init_params['AugImgDir'])
                delete_all(init_params['sem_seg_masks'])
                delete_all(init_params['panoptic_masks'])
            

            # previous model dir, CV for cross validation
            init_params['model_path'] = os.path.join(storage, models_dir, init_params['database'],
                                                     model_type, f'{percent_gt}_percent', args.data_set,
                                                     f"CV{init_params['CV']}", f'iter{exp - 1}', 'model_0024999.pth')
            if not os.path.exists(init_params['model_path']):                                         
                init_params['model_path'] = os.path.join(storage, models_dir, init_params['database'],
                                                        model_type, f'{percent_gt}_percent', args.data_set,
                                                        f"CV{init_params['CV']}", f'iter{exp - 1}', 'model_0019999.pth')
                if not os.path.exists(init_params['model_path']):                                         
                    init_params['model_path'] = os.path.join(storage, models_dir, init_params['database'],
                                                            model_type, f'{percent_gt}_percent', args.data_set,
                                                            f"CV{init_params['CV']}", f'iter{exp - 1}', 'model_0014999.pth')
            else:
                assert os.path.exists(init_params['model_path']), '{} is not available'.format(init_params['model_path'])
        
            
        if init_params['semi_supervised']:
            #labeled frames json will be used separately during training
            init_params['semi_frames'] = []
            semi_size = (len(init_params['semi_frames']))
            init_params['labeled_frames'] = init_params['semi_frames']

            #load all unlabeled frames: unlabeled windows are smaller than the labeled for AppleA
            #unalbeled json is used only to read frames in SSL
            json_path_unlabeled = f'{data_root}/instances_unlabeled_2021_{args.data_set}.json'
            #json_path_unlabeled = storage + '/SoftTeacher/data/{}/annotations/instances_unlabeled_2021.json'.format(args.database)
            _, unlabeled_frames = load_clasp_json(json_path_unlabeled, percent=100)
            print(f'total unlabeled: {len(unlabeled_frames)}\t semi gt: {semi_size}')
            init_params['unlabeled_frames'] = unlabeled_frames
            
            all_frames = unlabeled_frames
            assert len(all_frames)== len(set(unlabeled_frames))
            print(f'total frames for training: {len(all_frames)}')
            print(f'labeled frames anns json will read directly during training')
            #pdb.set_trace()

        else:
            all_frames = 'all frames ids' # will be removed in future

    return init_params, all_frames