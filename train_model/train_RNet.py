"""
yipengwa@usc.edu
"""
from train_model.mtcnn_model import R_Net
from train_model.train import train


if __name__ == '__main__':  # (batch_size
    model_name = 'MTCNN'
    model_path = '../save/{}_model/RNet'.format(model_name)
    img_dir = "../data/bbx_landmark"
    label_files = ["../data/bbx_landmark/landmark_24.txt",
                   "../data/bbx_landmark/part_24.txt",
                   "../data/bbx_landmark/pos_24.txt"]
    end_epoch = 22
    display = 200
    lr = 0.000001
    train(net_factory=R_Net,
          model_path=model_path,
          img_dir=img_dir,
          label_files=label_files,
          end_epoch=end_epoch,
          display=display,
          base_lr=lr)