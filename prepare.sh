#!/usr/bin/env bash
#cd prepare_data
#python prepare_data.py --mode train --img_dir ../data/bbx --label_file ../data/bbx/trainImageList.txt --img_size 48 --aug_scale 5
#python prepare_data.py --mode val --img_dir ../data/bbx --label_file ../data/bbx/testImageList.txt --img_size 48 --aug_scale 1
#cd ..

cd prepare_data
python prepare_data.py --mode train --img_dir ../data/human_bbx --label_file ../data/anime_bbx/anime_face_train.txt --img_size 48 --aug_scale 5
python prepare_data.py --mode val --img_dir ../data/human_bbx --label_file ../data/anime_bbx/anime_face_val.txt --img_size 48 --aug_scale 1
cd ..