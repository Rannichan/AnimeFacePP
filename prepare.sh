#!/usr/bin/env bash
cd prepare_data
python prepare_data.py --mode train --img_dir ../data/bbx --label_file ../data/bbx/trainImageList.txt --img_size 48 --aug_scale 5
python prepare_data.py --mode val --img_dir ../data/bbx --label_file ../data/bbx/testImageList.txt --img_size 48 --aug_scale 1
cd ..
