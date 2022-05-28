## CenterNet:Object as Points, Implementation of CenterNet Object Detection Model in Tensorflow2

---

## Table of Contents
1. [Update Top News ]
2. [Performance](#Performance)
3. [Required Environment](#Required-Environment)
4. [Notes on Attention](#Notes-on-Attention)
5. [Download](#Download)
6. [Training Steps  : How2train](#Training-Steps)
7. [Prediction Step : How2predict](#Prediction-Step)
8. [Assessment Step : How2val](#Assessment-Step)
9. [Reference](#Reference)

## Top News
**`2022-04`**:**There were significant updates, supporting step, COS learning rate drop method, supporting ADAM, SGD optimizer selection, supporting learning rate adjustment according to Batch_size adaptation, and additional picture cutting.Supports multi-GPU training, adds calculation of target by type, and adds heatmap.**  
The original warehouse address in the BiliBili video is: https://github.com/bubbliiiing/centernet-tf2/tree/bilibili

**`2021-10`**:**Large updates were made, annotations were added, adjustable parameters were added, code modules were modified, fps, video forecasting, bulk forecasting, etc.。**   

## Performance
| training dataset | weight file name | test dataset | input picture size | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12 | [centernet_resnet50_voc.h5](https://github.com/bubbliiiing/centernet-tf2/releases/download/v1.0/centernet_resnet50_voc.h5) | VOC-Test07 | 512x512 | - | 77.1
| COCO-Train2017 | [centernet_hourglass_coco.h5](https://github.com/bubbliiiing/centernet-tf2/releases/download/v1.0/centernet_hourglass_coco.h5) | COCO-Val2017 | 512x512 | 39.0 | 57.6 

## Required Environment
tensorflow-gpu==2.2.0  
No extra keras is required because there is already a keras section in tensorflow2

## Notes on Attention
The `centet_resnet50_voc.h5` in code is trained using the ***voc dataset***.
The `centet_hourglass_coco.h5` in the code is trained using the ***VOC dataset***.
**Be careful not to use Chinese labels, and don't leave any spaces in the folder!**     
**Be sure to create a new `txt` file under your *model_data* before training. Enter the classes you want to detect in the file. Point classes to the file at `train.py`**。     

## Download 
The weights for the training requirements of `centernet_resnet50_voc.h5`, `centernet_hourglass_coco.h5`, and the backbone can be downloaded from the Baidu disk.
Link: https://pan.baidu.com/s/16f3YU8pC_r1Ow7J9XiEB8w     
Extraction code: rmr5    

`centernet_resnet50_voc.h5` is the weight of the voc dataset. 
`centernet_hourglass_coco.h5` is the weight of the Coco dataset.

The VOC dataset download address is as follows. It already contains training sets, test sets, verification sets (as with test sets) and does not need to be subdivided:
Link: https://pan.baidu.com/s/19Mw2u_df_nBzsC2lg20fQA   
Extraction code: j5ge   

## Training Steps

### a. Training VOC07+12 Dataset
1. Data Set Preparation
**This article uses VOC format for training, which is required, before training, You have to download VOC07+12 Dataset, uncompressed and placed in root directory**

2.data set processing   
Modify `voc_annotatino.py` and run the modified `voc_annotation.py` to generate `2007_train.txt` and `2007_val.txt` under root directory.

3. Begin network training.   
The default parameters for `train.py` are used to train the VOC dataset. Run `train.py` directly to begin training.

4. training result prediction  
  Training results prediction needs to use two files, `predict.py` and `predict.py`. We first need to go to `centernet.py` to modify `model_path` and `classes_path`. Both parameters must be modified.
  
**`model_path` points to the trained weight file in the Logs folder.
`classes_path` points to the txt file corresponding to the detection class.**
When the modificatoin is complete, you can run `predict.py` for prediction. It can be detected by entering the path of the picture after running.

### b. Train your own dataset
1. Data set preparation
** This article uses VOCFormat training, need to make your own dataset before training,**
Place label files before training. Annotation under `VOC2007` folder under `VOCdevkit` folder of your environment.
Place picture files before trainingJPEG Images under VOC 2007 folder under VOC devkit folder of your environment.

2. Data set processing
  After the data set has been placed, we need to use `voc_annotation.py` to get `2007_train.txt` and `2007_val.txt` for training.
  Modify the parameters in `voc_annotation.py`. The first training can only modify `classes_path`, which is used to point to the txtfile corresponding to the detection class.
  When you train your own dataset, you can create your own `cls_classes.txt`, and write down the categories you want to distinguish.
  The contents of the `model_data/cls_classes.txt` file are as follows:
  ```python
cat
dog
...
```

Modify the `classes_path` in `voc_annotation.py` to correspond to `cls_classes.txt` and run `voc_annotation.py`.

3. Begin training network.  
**Most of the training parameters are available at `train.py`. You can read the notes carefully after downloading the library. The most important part of the training is still `classes_path` in `train.py`.**  
**`classes_path` is used to point to the txt file corresponding to the detection category, which is the same as the txt file in `voc_annotation.py`! Modify it to rain your own dataset**  
After modifying the `classes_path`, you can run `train.py` to start training. After training multiple epochs, the weights will be generated in the `logs` folder  

4. training result prediction  
Training results prediction needs to use two files, `predict.py` and `predict.py`. We first need to go to `centernet.py` to modify `model_path` and `classes_path`. Both parameters must be modified.

**`model_path` points to the trained weight file in the Logs folder.
`classes_path` points to the txt file corresponding to the detection class.**
When the modificatoin is complete, you can run `predict.py` for prediction. It can be detected by entering the path of the picture after running.

## Prediction Step
### a.Using Pre-Training Weights
1. Unzip the library after downloading, download the weight on the Baidu disk, put in model_data, run `predict.py`, enter
  ```python
img/street.jpg
```
2. You can set it up at `predict.py` to perform fps tests and video detection 
### b. Use one's own training weights
1. Follow the training procedures.  
2. In the `centernet.py` file, modify `model_path` and `classes_path` to correspond to the trained file in the following section: **`model_path` corresponds to the weight file under the `Logs` folder. `classes_path` is the classes of `model_path` corresponding to the classes**.
  ```python
_defaults = {
    #--------------------------------------------------------------------------#
    #  Make sure to modify `model_path` and `class_path` to make predictions using your own trained model!
    #  `model_path` point to the weight file under the `logs` folder and `classes_path` point to txt fiel under model_data
    #   If you experience a shape mismatch, pay attention to modification of `model_path` and `classes_path` parameters during training
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/centernet_resnet50_voc.h5',
    "classes_path"      : 'model_data/voc_classes.txt',
    #--------------------------------------------------------------------------#
    #   Used to select the backbone of the model to use
    #   resnet50, hourglass
    #--------------------------------------------------------------------------#
    "backbone"          : 'resnet50',
    #--------------------------------------------------------------------------#
    #   Enter the size of the picture
    #--------------------------------------------------------------------------#
    "input_shape"       : [512, 512],
    #--------------------------------------------------------------------------#
    #   Only the prediction box with a score greater than confidence will be retained.
    #--------------------------------------------------------------------------#
    "confidence"        : 0.3,
    #---------------------------------------------------------------------#
    #   non-maximum suppression `nms_iou` size
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #--------------------------------------------------------------------------#
    #   Non-maximum inhibition is optional based on the test results.
    #  It is recommended to set `True` when the backbone is resnet50 and `False` when the backbone is hour class
    #--------------------------------------------------------------------------#
    "nms"               : True,
    #---------------------------------------------------------------------#
    #   This variable is used to control whether the input image is resized without distortion using Letterbox_image.，
    #   After several test find it more effective to turn off letterbox_image direct resize
    #---------------------------------------------------------------------#
    "letterbox_image"   : False,
}
```
3. Run `predict.py` and enter  
  ```python
img/street.jpg
```
4. You can set it up at `predict.py` to perform fps tests and video detection.

## Assessment Steps 

### a. Assessment of VOC07+12 Test Set
1. This paper uses the VOC format for evaluation. VOC07+12 has divided the test sets and does not need to use `voc_annotation.py` to generate txts under the `ImageSet` folder.
2. Modify `model_path` and `classes_path` in `centernet.py`. **`model_path` points to the trained weight file in the Logs folder. `classes_path` points to a pair of detection classes pairtxt. **
3. Run `get_map.py` for evaluation. As a result, the evaluation results are saved in the `map_out` folder.

### b、Evaluate your own dataset
1. This article uses VOC format for evaluation.  
2. If the `voc_annotation.py` file has been run before training, the code will automatically divide the dataset into training sets, verification sets, and test sets. If you want to modify the ratio of the test set, you can modify the `trainval_percent` under the `voc_annotation.py` file. `trainval_percent` is used to specify the ratio (training set + validation set) to the test set. By default (training set + validation set): Test set = 9:1. 
3. After you divide the test set using `voc_annotation.py`, go to the `get_map.py` file and modify the `classes_path`, which is used to point to the txt file corresponding to the detection category, which is the same as the txt file in training. The dataset that evaluates itself must be modified.
4. Modify `model_path` and `classes_path` in `centernet.py`. **`model_path` points to the trained weight file in the Logs folder. `classes_path` points to the txt file corresponding to the detection class.**
5. Run `get_map.py` to get your evaluation results, which will be saved in the `map_out` folder

## Reference
https://github.com/xuannianz/keras-CenterNet      
https://github.com/see--/keras-centernet      
https://github.com/xingyizhou/CenterNet    
