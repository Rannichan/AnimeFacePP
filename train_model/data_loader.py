"""
yipengwa@usc.edu
"""
import os
import cv2
import tensorflow as tf
import random


def read_lable_file(label_file):
    # print('Read label file 2: the label file is ', label_file)
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
        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 12:
            bbox['xlefteye'] = float(info[2])
            bbox['ylefteye'] = float(info[3])
            bbox['xrighteye'] = float(info[4])
            bbox['yrighteye'] = float(info[5])
            bbox['xnose'] = float(info[6])
            bbox['ynose'] = float(info[7])
            bbox['xleftmouth'] = float(info[8])
            bbox['yleftmouth'] = float(info[9])
            bbox['xrightmouth'] = float(info[10])
            bbox['yrightmouth'] = float(info[11])
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
    read d
    :param img_dir:
    :param label_file:
    :return:
    """
    img_dir = str(img_dir, encoding="utf-8")
    labels = []
    for label_file in label_files:
        labels += read_lable_file(label_file)
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
        landmark = [bbox['xlefteye'], bbox['ylefteye'],
                    bbox['xrighteye'], bbox['yrighteye'],
                    bbox['xnose'], bbox['ynose'],
                    bbox['xleftmouth'], bbox['yleftmouth'],
                    bbox['xrightmouth'], bbox['yrightmouth']]
        yield (image_data, class_label, roi, landmark)


def batch_generator(img_dir, label_files, batch_size, net, shuffle=False):
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
    if net == "PNet": size = 12
    elif net == "RNet": size = 24
    elif net == "ONet": size = 48
    else:
        size = 0
        exit('Net type error')

    types = (tf.int32, tf.int32, tf.float32, tf.float32)
    shapes = ((size,size,None), (), (4,), (10,))

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
