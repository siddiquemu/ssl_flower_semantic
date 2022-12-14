#!/usr/bin/env bash
set -x

TYPE=$1 # SL or SSL or semi
ITER=$2
PERCENT=$3
GPUS=$4
GPU_ID=$0
DATASET=$5
CV=$6
PORT=${PORT:-29500}

#for ITER in ssl_iter; do   bash train_semi_iters_flower_2gpus.sh model_type ${ITER} --label_percent GPUS data_set CV; done
#for ITER in 1; do   bash train_semi_iters_flower_2gpus.sh SSL ${ITER} 100 2 AppleA 2; done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [[ ${TYPE} == 'SL' ]]; then
    python tools/trainAppleA_new.py --CV ${CV} --ssl_iter ${ITER} --lambda_sem 0.8 --gpu_id 0 --database flower --data_set ${DATASET} --label_percent ${PERCENT}\
           --model_type ${TYPE} --pretrained 1
else
    #Run evaluation on the test/validation set
    # to evaluate already trained model
    # python utils/sliding_windows_RGR.py --CV ${CV} --isTrain 1 --data_set ${DATASET} --ssl_iter ${ITER-1} --isLocal 1 --gpu_id 1
    #prepare sliding window frames to apply pseudo-labels method
    # python dataset/data_aug_train_CV.py --CV ${CV} --dataset ${DATASET}
    # generate pseudo labels using 3*N processes in N GPUS
    python tools/pseudo_labels_panoptic_flower_2gpus.py --CV ${CV} --ssl_iter ${ITER} --database flower --data_set ${DATASET} --label_percent ${PERCENT} \
             --number_gpus ${GPUS} --model_type ${TYPE}
    # # train using computed pseudo labels
    python tools/trainAppleA_new.py --number_gpus ${GPUS} --CV ${CV} --ssl_iter ${ITER} --lambda_sem 0.8 --gpu_id 0 --database flower --data_set ${DATASET} \
    --label_percent ${PERCENT} --model_type ${TYPE}
fi