#!/usr/bin/env bash
cd prepare_data
python prepare_data.py --mode train_bbox --net RNet --img_dir ../data/bbx_landmark --label_file ../data/bbx_landmark/trainImageList.txt
python prepare_data.py --mode train_landmark --net RNet --img_dir ../data/bbx_landmark --label_file ../data/bbx_landmark/trainImageList.txt
cd ..
