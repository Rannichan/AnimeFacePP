#!/usr/bin/env bash
cd train_model
source activate auggan
python train_manga.py \
--img_dir ../data/bbx \
--label_files ../data/bbx/pos_48_5.txt\ ../data/bbx/neg_48_5.txt \
--val_label_files ../data/bbx/pos_48_1.txt\ ../data/bbx/neg_48_1.txt \
--epoch 25 \
--display 200 \
--lr 1e-6