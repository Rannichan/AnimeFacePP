"""
yipengwa@usc.edu
"""
import os
import cv2
import random
import numpy as np
import tensorflow as tf


def read_label_file(label_file):
    imagelist = open(label_file, 'r')
    data = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_unit = dict()
        bbox = dict()
        data_unit['filename'] = info[0].replace('\\', '/')
        data_unit['label'] = int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        data_unit['bbox'] = bbox
        data.append(data_unit)
    return data


def read_label_file2(label_file):
    imagelist = open(label_file, 'r')
    data = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_unit = dict()
        bbox = dict()
        data_unit['filename'] = info[0].replace('\\', '/')
        bbox['xmin'] = float(info[1])
        bbox['xmax'] = float(info[2])
        bbox['ymin'] = float(info[3])
        bbox['ymax'] = float(info[4])
        bbox['xlefteye'] = float(info[5])
        bbox['ylefteye'] = float(info[6])
        bbox['xrighteye'] = float(info[7])
        bbox['yrighteye'] = float(info[8])
        bbox['xnose'] = float(info[9])
        bbox['ynose'] = float(info[10])
        bbox['xleftmouth'] = float(info[11])
        bbox['yleftmouth'] = float(info[12])
        bbox['xrightmouth'] = float(info[13])
        bbox['yrightmouth'] = float(info[14])
        data_unit['bbox'] = bbox
        data.append(data_unit)
    return data


def read_image_file(image_file):
    image = cv2.imread(image_file)
    # image_data = image.tostring()
    height = image.shape[0]
    width = image.shape[1]
    # return string data and initial height and width of the image
    return image, height, width


def data_generator(img_dir, label_files, shuffle=True):
    """

    :param img_dir:
    :param label_file:
    :return:
    """
    img_dir = str(img_dir, encoding="utf-8")
    labels = []
    for label_file in label_files:
        labels += read_label_file(label_file)
    if shuffle:
        random.shuffle(labels)

    for i, data_unit in enumerate(labels):
        filename = data_unit['filename']
        img_path = os.path.join(img_dir, filename)
        image_data, height, width = read_image_file(img_path)
        class_label = data_unit['label']
        bbox = data_unit['bbox']
        roi = [bbox['xmin'], bbox['ymin'],
               bbox['xmax'], bbox['ymax']]
        yield (image_data, class_label, roi)


def batch_generator(img_dir, label_files, batch_size, img_size, shuffle=False):
    '''
    Batchify data
    sents: list of sents
    vocab_path: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (batch_size, seqlen)
        x_seqlens: int32 tensor. (batch_size,)
        sents: str tensor. (batch_size,)
    '''
    types = (tf.int32, tf.int32, tf.float32)
    shapes = ((img_size, img_size, None), (), (4,))

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types=types,
        output_shapes=shapes,
        args=(img_dir, label_files))

    if shuffle:  # for training
        dataset = dataset.shuffle(128 * batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    return dataset


def load_all(img_dir, label_files, img_size):
    """
    load all data at once for validation
    :param img_dir:
    :param label_file:
    :param img_size:
    :return:
    """
    labels = []
    for label_file in label_files:
        labels += read_label_file(label_file)
    name_array = []
    image_array = []
    label_array = []
    bbox_array = []
    for data_unit in labels:
        filename = data_unit["filename"]
        bbox = data_unit["bbox"]
        label = data_unit["label"]

        # read image
        img_path = os.path.join(img_dir, filename).replace('\\', '/')
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        if height != img_size or width != img_size: continue

        # # read normalized bounding box
        xmin, ymin, xmax, ymax = float(bbox["xmin"]), float(bbox["ymin"]), float(bbox["xmax"]), float(bbox["ymax"])
        bbox = [xmin, ymin, xmax, ymax]
        # box_normalized = [float(xmin)/width, float(ymin)/height, float(xmax)/width, float(ymax)/height]

        # resized_im = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        name_array.append(filename)
        image_array.append(img)
        label_array.append(label)
        bbox_array.append(bbox)

    return np.array(name_array), np.array(image_array), np.array(label_array), np.array(bbox_array)
