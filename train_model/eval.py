"""
yipengwa@usc.edu
"""
import os
import math
import random
import cv2
from datetime import datetime
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from mtcnn_model import R_Net
from MTCNN_config import config
from data_loader import batch_generator, load_all
from train_utils import random_flip_images, image_color_distort, pred_visualizer
from PIL import ImageDraw, Image


def eval(data_array, model_path, img_dir, regression=False):
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
    net = model_path.split('/')[-1]

    if net == 'PNet':
        image_size = 12
        radio_bbox_loss = 0.5; radio_landmark_loss = 0.5
    elif net == 'RNet':
        image_size = 24
        radio_bbox_loss = 0.5; radio_landmark_loss = 0.5
    else:
        image_size = 48
        radio_bbox_loss = 0.5; radio_landmark_loss = 1

    # saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(model_path)
        saver = tf.train.import_meta_graph(ckpt + '.meta')

        if ckpt is not None:
            print("Model found: {}".format(ckpt))
            saver.restore(sess, ckpt)
        else:
            print("Model not found")
            exit()

        # # for debugging
        # variable_names = [v.name for v in tf.all_variables()]
        # values = sess.run(variable_names)
        # for k, v in zip(variable_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        #     print(v)
        # exit()

        graph = tf.get_default_graph()
        input_image = graph.get_tensor_by_name('input_image:0')
        label = graph.get_tensor_by_name('label:0')
        bbox_target = graph.get_tensor_by_name('bbox_target:0')
        landmark_target = graph.get_tensor_by_name('landmark_target:0')

        # load data
        (name_array, image_array, label_array, bbox_array, landmark_array) = data_array
        feed_dict = {input_image: image_array,
                     label: label_array,
                     bbox_target: bbox_array,
                     landmark_target: landmark_array}

        bbox_pred_op = graph.get_tensor_by_name('bbox_fc/BiasAdd:0')
        landmark_pred_op = graph.get_tensor_by_name('landmark_fc/BiasAdd:0')
        bbox_loss_op = graph.get_tensor_by_name('Mean:0')
        landmark_loss_op = graph.get_tensor_by_name('Mean_1:0')
        L2_loss_op = graph.get_tensor_by_name('AddN:0')

        if regression:
            bbox_pred, landmark_pred = sess.run([bbox_pred_op, landmark_pred_op], feed_dict=feed_dict)
            # pred_output = os.path.join(img_dir, 'pred')
            pred_visualizer(img_dir, bbox_pred, landmark_pred, name_array)
        else:
            bbox_loss, landmark_loss, L2_loss = sess.run([bbox_loss_op, landmark_loss_op, L2_loss_op],
                                                         feed_dict=feed_dict)
            total_loss = radio_bbox_loss * bbox_loss + \
                         radio_landmark_loss * landmark_loss + \
                         L2_loss
            return  bbox_loss, landmark_loss, L2_loss, total_loss


if __name__ == "__main__":
    data_array = load_all('../data/bbx_landmark', '../data/bbx_landmark/testImageList.txt', 'RNet')
    output = eval(data_array=data_array,
                  model_path='../save/MTCNN_model/RNet',
                  img_dir='../data/bbx_landmark',
                  regression=True)

    print(output)
