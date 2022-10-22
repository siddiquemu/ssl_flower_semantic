# Self-supervised Learning for Panoptic Segmentation of Multiple Fruit Flower Species.
(Accepted by IEEE Robotics and Automation Letters). Preprint available at [Paper](https://arxiv.org/abs/2209.04618)

# codeabse progress
- [ ] Preprocess raw data
- [ ] Apply data augmentation on the training data
- [ ] Train initial panoptic model using AppleA_train
- [x] Preapre train/test unlabeled datsets for multiple run
- [ ] Generate panoptic pseudo-labels for finetuning the initial model
- [ ] Train iteratively using pseudo labels
- [x] Evaluate the model

### Requirements: ###
* Detectron2
* Python 3.8
* Pytorch 1.9
* CUDA 10.2
* Pycocotools 2.0

### Installation ###

3. This codebase is heavily based on [Detectron2](https://github.com/facebookresearch/detectron2) an instacne segmentation refinement method [RGR](https://bitbucket.org/phil_dias/rgr-public/src/master/). Install baoth and keep RGR in the root folder
```./ssl_flower_semantic/
```

1. clone this repository and go to root folder
```python
https://github.com/siddiquemu/ssl_flower_semantic.git
cd ssl_flower_semantic
```
2. create a environment
```python
pip install -r det2_requirements.yml
```


### Data Preprocessing ###
1. Download the raw data from (will be updated soon). The folder structure will be
```
./dataseet/raw_data/
├── imgs
│   ├── AppleA
│   ├── AppleA_train
│   ├── AppleB
│   ├── Peach
│   └── Pear
└── labels
    ├── AppleA
    │   └── gt_frames
    ├── AppleA_train
    │   └── gt_frames
    ├── AppleB
    │   └── gt_frames
    ├── Peach
    │   └── gt_frames
    └── Pear
        └── gt_frames
```
2. run the following script from root to generate the train/test split for CV experiments
```
python ./dataset/data_aug_train_CV.py
```

For example the data folder structure for CV=1 in data root directory will be as follows
```
./dataset/ssl_data/
```
```
├── AppleA
│   └── CV1
│       ├── test_imgs
│       └── train_imgs
├── AppleB
│   └── CV1
│       ├── test_imgs
│       └── train_imgs
├── Peach
│   └── CV1
│       ├── test_imgs
│       └── train_imgs
└── Pear
    └── CV1
        ├── test_imgs
        └── train_imgs
```

### Test ###
1. To test the models, download CV1 models from ....
```
├── SL
│   └── AppleA_train
├── SSL
│   ├── AppleA
│   │   └── CV1
│   │       └── iter3
│   ├── AppleB
│   │   └── CV1
│   │       └── iter3
│   ├── Peach
│   │   └── CV1
│   │       └── iter6
│   └── Pear
│       └── CV1
│           └── iter3
└── SSL_RGR
    ├── AppleA
    │   └── CV1
    │       └── iter2
    ├── AppleB
    │   └── CV1
    │       └── iter2
    ├── Peach
    │   └── CV1
    │       └── iter6
    └── Pear
        └── CV1
            └── iter3
```
2. run the following script to evaluate the CV models

```
python utils/sliding_windows_RGR.py --CV 1 --data_set AppleB --ssl_iter 3 --isLocal 1 --gpu_id 0 --model_type SSL
```

### Train ###
To train the SL model using augmented AppleA train set:
1. run the following script from root to prepare training data
```
python ./dataset/data_aug_train.py --dataset AppleA_train
```
2.  go to root directory and run

```
for ITER in ssl_iter; do   bash train_semi_iters_flower_2gpus.sh model_type ${ITER} --label_percent GPUS data_set CV; done
for ITER in 1; do   bash train_semi_iters_flower_2gpus.sh SSL ${ITER} 100 2 AppleA_train 2; done
```
3. To train the SSL model using AppleA trained model and the unlabeled data: go to root directory and run

```
for ITER in 1 2 3; do   bash train_semi_iters_flower_2gpus.sh SSL ${ITER} 100 2 AppleA_train 2; done

```
4. To train the pretrained model using the unlabeled data: go to root directory and run

```
for ITER in 1; do   bash train_semi_iters_flower_2gpus.sh SSL ${ITER} 100 2 AppleA 2; done
```
### Citing ssl_flower_semantic ###
If you find this work helpful in your research, please cite using the following bibtex
```
@article{siddique2022self,
  title={Self-supervised Learning for Panoptic Segmentation of Multiple Fruit Flower Species},
  author={Siddique, Abubakar and Tabb, Amy and Medeiros, Henry},
  journal={arXiv preprint arXiv:2209.04618},
  year={2022}
}
```