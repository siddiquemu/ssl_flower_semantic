import cv2
import glob
import os

AppleA_147 = glob.glob('/media/6TB_local/tracking_wo_bnw/data/flower/AppleA/FlowerImages/*JPG')
ApplA_train_gts = glob.glob('/media/6TB_local/RAL_code/ssl_flower_semantic/dataset/raw_data/labels/AppleA_train/gt_frames/*')

print(f'all AppleA (train+test): {len(AppleA_147)}, gts: {len(ApplA_train_gts)}')

for imname in AppleA_147:
    ApplA_train_gts
    fr = int(os.path.basename(imname).split('.')[0].split('IMG_')[-1])
    gt_base = f'/media/6TB_local/RAL_code/ssl_flower_semantic/dataset/raw_data/labels/AppleA_train/gt_frames/{fr}.png'
    
    if gt_base in ApplA_train_gts:
        print(os.path.basename(gt_base), os.path.basename(imname))
        im = cv2.imread(imname)
        cv2.imwrite(f'/media/6TB_local/RAL_code/ssl_flower_semantic/dataset/raw_data/imgs/AppleA_train/{os.path.basename(imname)}', im)