"""
yipengwa@usc.edu
"""
import os
import tensorflow as tf
from data_loader import load_all
from train_utils import pred_visualizer
from PIL import ImageDraw, Image
from model import Manga_Net


def eval(data_array, model_path, img_dir, image_size=48, regression=False):
    """
    If regression is False, compute bbox loss and landmark loss;
    If regression is True, do bbox regression and landmark regression.
    :param net_factory:
    :param model_path:
    :param img_dir:
    :param label_files:
    :param regression:
    :return:
    """
    radio_bbox_loss = 0.7
    radio_cls_loss = 0.3

    # define placeholder
    input_image = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.int32, shape=[None], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[None, 4], name='bbox_target')
    cls_loss_op, bbox_loss_op, L2_loss_op, cls_pred_op, bbox_pred_op = Manga_Net(inputs=input_image,
                                                                                 label=label,
                                                                                 bbox_target=bbox_target)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(model_path)
        if ckpt is not None:
            print("Model found: {}".format(ckpt))
            saver.restore(sess, ckpt)
        else:
            print("Model not found")
            exit()
        # saver.restore(sess, "../save/Manga_Net_model/manga_net_epoch18")

        # # for debugging
        # variable_names = [v.name for v in tf.all_variables()]
        # values = sess.run(variable_names)
        # for k, v in zip(variable_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        #     print(v)
        # exit()

        # load data
        (name_array, image_array, label_array, bbox_array) = data_array
        feed_dict = {input_image: image_array,
                     label: label_array,
                     bbox_target: bbox_array}

        if regression:
            cls_pred, bbox_pred = sess.run([cls_pred_op, bbox_pred_op], feed_dict=feed_dict)
            pred_visualizer(img_dir, cls_pred, bbox_pred, name_array)
            acc = sum(cls_pred)/len(cls_pred)
            return acc, len(cls_pred)
        else:
            cls_loss, bbox_loss, L2_loss = sess.run([cls_loss_op, bbox_loss_op, L2_loss_op],
                                                    feed_dict=feed_dict)
            total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * bbox_loss + L2_loss
            return cls_loss, bbox_loss, L2_loss, total_loss


if __name__ == "__main__":
    label_files = ["../data/bbx/pos_48_1.txt"] # , "../data/bbx/neg_48_1.txt"
    data_array = load_all('../data/bbx', label_files, 48)
    output = eval(data_array=data_array,
                  model_path='../save/Manga_Net_model',
                  img_dir='../data/bbx',
                  regression=True)
    print(output)
