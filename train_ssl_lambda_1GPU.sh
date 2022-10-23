#!/usr/bin/env bash
set -x

TYPE=$1 # SL or SSL or semi
ITER=$2
PERCENT=$3
GPUS=$4
GPU_ID=$0
DATASET=$5
PORT=${PORT:-29500}


#for ITER in 1; do   bash train_semi_iters_flower_2gpus.sh semi ${ITER} 100 2 AppleA; done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [[ ${TYPE} == 'SL' ]]; then
    python trainAppleA_new.py --ssl_iter ${ITER} --lambda_sem 0.8 --gpu_id 0 --database flower --data_set ${DATASET} --label_percent ${PERCENT}\
          --working_dir /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62 --model_type ${TYPE}
else
    # semi
    # generate pseudo labels using 3*N processes in N GPUS
    # python pseudo_labels_panoptic_flower_2gpus.py --ssl_iter ${ITER} --database flower --data_set ${DATASET} --label_percent ${PERCENT} \
    #          --working_dir /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62
    # train using computed pseudo labels
    for lambda_s in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.9; # 
        do python trainAppleA_lambda.py --ssl_iter ${ITER} --lambda_sem ${lambda_s} --gpu_id 0 --database flower --data_set ${DATASET} --label_percent ${PERCENT}\
            --working_dir /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62 --model_type ${TYPE}; 
        done

fi