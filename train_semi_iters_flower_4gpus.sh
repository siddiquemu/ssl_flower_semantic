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

#for ITER in ssl_iter; do   bash train_semi_iters_flower_2gpus.sh model_type ${ITER} 
#--label_percent GPUS data_set CV; done
#for ITER in 1; do   bash train_semi_iters_flower_2gpus.sh SSL ${ITER} 100 2 AppleA 2; done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [[ ${TYPE} == 'SL' ]]; then
    python trainAppleA_new.py --CV ${CV} --ssl_iter ${ITER} --lambda_sem 0.8 --gpu_id 0 --database flower --data_set ${DATASET} --label_percent ${PERCENT}\
          --working_dir /media/siddique/6TB --model_type ${TYPE}
else
    #Run evaluation on the test/validation set
    #python sliding_windows_RGR.py --CV ${CV} --isTrain 1 --data_set ${DATASET} --ssl_iter ${ITER} --isLocal 0 --gpu_id 1 --pretrained 1

    # generate pseudo labels using 3*N processes in N GPUS
    python pseudo_labels_panoptic_flower.py --CV ${CV} --ssl_iter ${ITER} --database flower --data_set ${DATASET} --label_percent ${PERCENT} \
             --number_gpus ${GPUS} --working_dir /media/siddique/6TB5 
    # train using computed pseudo labels
    python trainAppleA_new.py --number_gpus ${GPUS} --CV ${CV} --ssl_iter ${ITER} --lambda_sem 0.8 --gpu_id 0 --database flower --data_set ${DATASET} \
    --label_percent ${PERCENT} --working_dir /media/siddique/6TB5 --model_type ${TYPE}

fi

