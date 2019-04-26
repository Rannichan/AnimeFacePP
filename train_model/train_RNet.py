"""
yipengwa@usc.edu
"""
import argparse
from mtcnn_model import R_Net
from train import train


if __name__ == '__main__':  # (batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="../data/bbx_landmark", help="image directory", type=str)
    parser.add_argument("--label_file", default="../data/bbx_landmark/landmark_24.txt ../data/bbx_landmark/part_24.txt ../data/bbx_landmark/pos_24.txt",
                        help="label files for training image", type=str)
    parser.add_argument("--val_label_file", default="../data/bbx_landmark/testImageList.txt",
                        help="label file for validation image", type=str)
    parser.add_argument("--epoch", default=22, help="total training epoches", type=int)
    parser.add_argument("--display", default=200, help="how often do we do the validation during training", type=int)
    parser.add_argument("--lr", default=0.000001, help="base learning rate", type=float)
    args = parser.parse_args()
    img_dir = args.img_dir
    label_files = args.label_file.split()
    val_label_file = args.val_label_file
    end_epoch = args.epoch
    display = args.display
    lr = args.lr
    model_name = 'MTCNN'
    model_path = '../save/{}_model/RNet'.format(model_name)

    # print(img_dir); print(label_files); print(val_label_file)
    # print(end_epoch); print(display); print(lr); print(model_path)
    # exit()

    train(net_factory=R_Net,
          model_path=model_path,
          img_dir=img_dir,
          label_files=label_files,
          val_label_file=val_label_file,
          end_epoch=end_epoch,
          display=display,
          base_lr=lr)