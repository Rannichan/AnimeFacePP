"""
yipengwa@usc.edu
"""
import os
import math
import random
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from MTCNN_config import config


def train_model(base_lr, loss, data_num):
    """
    train model
    :param base_lr: base learning rate
    :param loss: loss
    :param data_num: number of training data
    :return:
    train_op, lr_op
    """
    warmup_factor = 5
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False, name="global_step")
    # LR_EPOCH [8,14]
    # boundaried [num_batch,num_batch]
    num_train_batches = math.ceil(data_num / config.BATCH_SIZE)
    boundaries = [int(epoch * num_train_batches) for epoch in config.LR_EPOCH]
    # lr_values[0.01, 0.001, 0.0001, 0.00001]
    lr_values = [base_lr, base_lr * warmup_factor] + [base_lr * (lr_factor ** x) for x in
                                                      range(1, len(config.LR_EPOCH))]
    # control learning rate
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)
    return train_op, lr_op


# all mini-batch mirror
def random_flip_images(image_batch, label_batch, landmark_batch):
    # mirror
    if random.choice([0, 1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch == -2)[0]
        flipposindexes = np.where(label_batch == 1)[0]
        # only flip
        flipindexes = np.concatenate((fliplandmarkindexes, flipposindexes))
        # random flip
        for i in flipindexes:
            cv2.flip(image_batch[i], 1, image_batch[i])

            # pay attention: flip landmark
        for i in fliplandmarkindexes:
            landmark_ = landmark_batch[i].reshape((-1, 2))
            landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]  # left eye<->right eye
            landmark_[[3, 4]] = landmark_[[4, 3]]  # left mouth<->right mouth
            landmark_batch[i] = landmark_.ravel()

    return image_batch, landmark_batch


def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs, max_delta=0.2)
    inputs = tf.image.random_saturation(inputs, lower=0.5, upper=1.5)
    return inputs


def pred_visualizer(img_dir, bbox_pred, landmark_pred, name_array):
    pred_output = os.path.join(img_dir, 'pred')
    if not os.path.exists(pred_output): os.makedirs(pred_output)
    for file_name, bbox, landmark in zip(name_array, bbox_pred, landmark_pred):
        img_file = os.path.join(img_dir, file_name)
        img = Image.open(img_file)
        size = img.size[0]
        draw = ImageDraw.Draw(img)
        draw.rectangle([(bbox[0] * size, bbox[1] * size),
                        (bbox[2] * size, bbox[3] * size)],
                       outline='yellow')
        for i in range(5):
            draw.rectangle([(landmark[2 * i] * size - 2, landmark[2 * i + 1] * size - 2),
                            (landmark[2 * i] * size + 2, landmark[2 * i + 1] * size + 2)],
                           outline='red')
        img.save(os.path.join(pred_output, file_name.split('/')[-1]))



