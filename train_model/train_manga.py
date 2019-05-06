"""
yipengwa@usc.edu
"""
import argparse
from model import Manga_Net
from train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="../data/bbx", help="image directory", type=str)
    parser.add_argument("--label_files", default="../data/bbx/pos_48.txt ../data/bbx/neg_48.txt",
                        help="label file for training image", type=str)
    parser.add_argument("--val_label_files", default="../data/bbx/val.txt",
                        help="label file for validation image", type=str)
    parser.add_argument("--epoch", default=22, help="total training epoches", type=int)
    parser.add_argument("--display", default=200, help="how often do we do the validation during training", type=int)
    parser.add_argument("--lr", default=0.000001, help="base learning rate", type=float)
    args = parser.parse_args()
    img_dir = args.img_dir
    label_files = args.label_files.split()
    val_label_files = args.val_label_files.split()
    end_epoch = args.epoch
    display = args.display
    lr = args.lr
    model_name = 'Manga_Net'
    model_path = '../save/{}_model'.format(model_name)

    # print(img_dir); print(label_files); print(val_label_file)
    # print(end_epoch); print(display); print(lr); print(model_path)
    # exit()

    train(net_factory=Manga_Net,
          model_path=model_path,
          img_dir=img_dir,
          label_files=label_files,
          val_label_files=val_label_files,
          end_epoch=end_epoch,
          display=display,
          base_lr=lr)
