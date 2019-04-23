[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)
# AnimeFacePP
Animation face keypoints detector


# Prepare training/evaluation data
Put your original image data, together with its label file, under 'data' directory. (like 'lfw_5590' and testImageList.txt)

Under 'prepare_data' directory, modify the code of prepare_data.py for your data path, then run it to get training/evaluation data.
```angular2html
cd prepare_data
python prepare_data.py
``` 

# Start training
Under 'train_model' directory, modify the code of train_RNet.py for your data path, then run it to start training. 

Model will saved in 'save' directory.

You can use tensorboard to monitor the training process.
```angular2html
tensorboard --logdir=logs/RNet
``` 
