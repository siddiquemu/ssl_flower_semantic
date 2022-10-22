# Self-supervised Learning for Panoptic Segmentation of Multiple Fruit Flower Species.
(Accepted by IEEE Robotics and Automation Letters). Preprint available at [Paper](https://arxiv.org/abs/2209.04618)

### Requirements: ###
* Detectron2
* Python 3.8
* Pytorch 1.9
* CUDA 10.2
* Pycocotools 2.0

### Installation ###

1. clone this repository and go to root folder
```python
https://github.com/siddiquemu/ssl_flower_semantic.git
cd ssl_flower_semantic
```
2. create a environment
```python
pip install -r det2_requirements.yml
```

3. setup instacne segmentation refinement method envrionment [RGR](https://bitbucket.org/phil_dias/rgr-public/src/master/) and keep it in the root folder
```./ssl_flower_semantic/
```


### Data Preprocessing ###
1. Download the raw data from ....
2. run the following script from root to generated the train/test split for CV experiments
```
python ./dataset/data_aug_train_CV.py
```

For example the data folder structure for CV=1 in root directory will be as follows
```
./dataset/ssl_data/
```
<br>
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

### Test ###
1. To test the models, download CV1 models from ....
```
        ├── SL
        |   |
        │   └── AppleA_train
        └── SSL
            |
            ├── AppleA
            |   |
            │   └── CV1
            |       |
            │       └── iter3
            ├── AppleB
            |   |
            │   └── CV1
            |       |
            │       └── iter3
            ├── Peach
            |   |
            │   └── CV1
            |       |
            │       └── iter6
            └── Pear
                |
                └── CV1
                    |
                    └── iter3
```
2. run the following script to evaluate the CV models

```
python sliding_windows_RGR.py --CV 1 --data_set Pear --ssl_iter 5 --isLocal 1 --gpu_id 1
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