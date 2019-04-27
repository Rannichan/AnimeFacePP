#!/usr/bin/env bash
cd train_model
source activate auggan
python train_RNet.py \
--img_dir ../data/bbx_landmark \
--label_file ../data/bbx_landmark/landmark_24.txt\ ../data/bbx_landmark/part_24.txt\ ../data/bbx_landmark/pos_24.txt \
--val_label_file ../data/bbx_landmark/testImageList.txt \
--epoch 22 \
--display 200 \
--lr 1e-6