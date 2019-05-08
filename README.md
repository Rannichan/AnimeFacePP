[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)
# AnimeFacePP
Animation face keypoints detector


## Prepare training data
* Make sure the label files at the right palce in data folder
```angular2html
data/bbx/testImageList.txt   --> for test
data/bbx/trainImageList.txt  --> for train
```
* Make sure image file at the right palce in data folder
```angular2html
data/bbx/lfw_5590
data/bbx/net_7876
```
* Then use script to generate generate augmented training data (you can change parameters inside the script for customization)
```angular2html
bash prepare.sh
```

## Start training
* Run the training script to start training (you can change parameters inside the script for customization)
```angular2html
bash train.sh
```
* log files will be save in "logs" folder
* tensorflow events will be saved in "logs/Manga_Net" folder. You can use tensorboard to monitor the training process.
```angular2html
tensorboard --logdir=logs/Manga_Net
``` 
* model will be saved in "save" folder

## Evaluation
Run following script to do evaluation
```angular2html
cd train_model
python eval.py
```
Prediction accuracy will be printed out on screen;
Prediction result will be saved in the directory
```angular2html
data/bbx/pred
```