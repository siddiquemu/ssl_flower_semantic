# Self-supervised Learning for Flower Segmentation.

### Requirements: ###
* Detectron2
* Python 3.8
* Pytorch 1.9
* CUDA 10.2
* Pycocotools 2.0

### Installation ###

1. clone this repository and go to root folder
```python
git clone https://Siddiquemu@bitbucket.org/Siddiquemu/ssl_flower_panoptic.git
cd ssl_flower_panoptic
```
2. create a environment
```python
pip install -r det2_requirements.yml
```

3. setup instacne segmentation refinement method envrionment [RGR](https://bitbucket.org/phil_dias/rgr-public/src/master/)

5. run the following script to prepare the repo for training/testing

<!-- ```python
bash tools/prepare_scripts.sh
``` -->

### Data Preprocessing ###
1. Download the train/test data and trained models from ... and move the dowloaded data and models folder into the root directory


### Test ###
1. To test the models

```
python sliding_windows_RGR.py --CV 1 --data_set Pear --ssl_iter 5 --isLocal 1 --gpu_id 1
```

### Train ###
1. To train the SL model using AppleA train set: go to root directory and run

```
for ITER in ssl_iter; do   bash train_semi_iters_flower_2gpus.sh model_type ${ITER} --label_percent GPUS data_set CV; done
for ITER in 1; do   bash train_semi_iters_flower_2gpus.sh SSL ${ITER} 100 2 AppleA_train 2; done
```
2. To train the SSL model using AppleA trained model and the unlabeled data: go to root directory and run

```
for ITER in 1; do   bash train_semi_iters_flower_2gpus.sh SSL ${ITER} 100 2 AppleA_train 2; done

```
3. To train the pretrained model using the unlabeled data: go to root directory and run using --pretrained 1 in bash train_semi_iters_flower_2gpus.sh script

```
for ITER in 1; do   bash train_semi_iters_flower_2gpus.sh SSL ${ITER} 100 2 AppleA_train 2; done
```
### Citing ssl_flower ###