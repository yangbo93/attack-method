# RT-DIM
This repository contains code to reproduce results from the paper:
**Random Transformation of Image Brightness for Adversarial Attack**

## REQUIREMENTS
- Environment Anaconda
- Python 3.7
- Tensorflow 1.14
- Numpy 1.18.1 
- cv2 4.4.0.42
- scipy 1.4.1

## Method
we propose a new attack method based on data augmentation that randomly transforms the brightness of the input image at each iteration in the attack process to alleviate overfitting and generate adversarial examples with more transferability. We summarize our algorithm in [Random Transformation of Image Brightness for Adversarial Attack].

### Dataset
We use a subset of ImageNet validation set containing 1000 images, most of which are correctly classified by those models.

### Models
We use the ensemble of seven models in our submission, many of which are adversarially trained models. The models can be downloaded in (https://github.com/tensorflow/models/tree/master/research/slim, https://github.com/tensorflow/models/tree/master/research/adv imagenet models).










