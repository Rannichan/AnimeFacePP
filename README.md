[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)
# AnimeFacePP
Animation face keypoints detector


## Prepare training data
* Make sure the label files at the right palce in data folder
```angular2html
data/bbx_landmark/testImageList.txt
data/bbx_landmark/trainImageList.txt
```
* Make sure image file at the right palce in data folder
```angular2html
data/bbx_landmark/lfw_5590
data/bbx_landmark/net_7876
```
* Then use script to generate generate augmented training data
```angular2html
bash prepare.sh
```

## Start training
* Run the training script to start training
```angular2html
bash train.sh
```
* log files will be save in "logs" folder

* tensorflow events will be saved in "logs/RNet" folder. You can use tensorboard to monitor the training process.
```angular2html
tensorboard --logdir=logs/RNet
``` 
* model will be saved in "save" folder

## Evaluation
... to be completed