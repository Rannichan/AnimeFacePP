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
from MTCNN_config import config
from data_loader import batch_generator, load_all
from train_utils import train_model, random_flip_images, image_color_distort
from eval import eval
from PIL import ImageDraw, Image

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not os.path.exists('../logs'): os.makedirs('../logs')
fh = logging.FileHandler('../logs/log_{}.txt'.format(str(datetime.now())))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
filter = logging.Filter(name='root')
fh.setFormatter(formatter)
fh.addFilter(filter)
logger.addHandler(fh)


def train(net_factory, model_path, img_dir, label_files, val_label_file,
          end_epoch=20, display=100, base_lr=0.01):
    """
    train PNet/RNet/ONet
    :param net_factory: pre-defined model structure
    :param model_path: str, model path
    :param img_dir:
    :param label_file:
    :param end_epoch: int, total training epoch
    :param display:
    :param base_lr:
    :return:
    """
    if not os.path.exists(model_path): os.makedirs(model_path)
    net = model_path.split('/')[-1]
    logger.info("Current training net: {}".format(net))
    logger.info("Model will be saved in {}".format(model_path))
    num = 0
    for label_file in label_files:
        with open(label_file, 'r') as f:
            logger.info("Label file found: {}".format(label_file))
            num += len(f.readlines())
    logger.info("Size of the dataset is: ", num)

    # data loader for training
    train_batches = batch_generator(img_dir, label_files, config.BATCH_SIZE, net)
    iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
    image_batch, label_batch, bbox_batch, landmark_batch = iter.get_next()
    train_init_op = iter.make_initializer(train_batches)
    logger.info("Data loader created")

    # load data for validation
    val_data_array = load_all(img_dir, val_label_file, net)
        
    if net == 'PNet':
        image_size = 12
        radio_bbox_loss = 0.5; radio_landmark_loss = 0.5
    elif net == 'RNet':
        image_size = 24
        radio_bbox_loss = 0.5; radio_landmark_loss = 0.5
    else:
        image_size = 48
        radio_bbox_loss = 0.5; radio_landmark_loss = 1
    
    # define placeholder
    input_image = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[None], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[None, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32,shape=[None, 10],name='landmark_target')
    logger.info("Input tensor placeholder defined")

    # get loss and accuracy
    input_image = image_color_distort(input_image)
    bbox_loss_op, landmark_loss_op, L2_loss_op = net_factory(inputs=input_image,
                                                             label=label,
                                                             bbox_target=bbox_target,
                                                             landmark_target=landmark_target,
                                                             training=True)
    print("name: ", bbox_loss_op.name, landmark_loss_op.name, L2_loss_op.name)
    total_loss_op  = radio_bbox_loss * bbox_loss_op + \
                     radio_landmark_loss * landmark_loss_op + \
                     L2_loss_op
    train_op, lr_op = train_model(base_lr, total_loss_op, num)
    logger.info("Loss operation and training operation defined")

    # # for debugging
    # for var in tf.all_variables():
    #     print(var)
    # exit()

    # init
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=0)
    sess = tf.Session()
    sess.run(train_init_op)
    sess.run(init)
    logger.info("Graph initialized")

    #visualize some variables
    tf.summary.scalar("bbox_loss", bbox_loss_op)
    tf.summary.scalar("landmark_loss", landmark_loss_op)
    tf.summary.scalar("total_loss", total_loss_op)
    summary_op = tf.summary.merge_all()
    logs_dir = "../logs/{}".format(net)
    if os.path.exists(logs_dir) == False:
        os.makedirs(logs_dir)
    writer = tf.summary.FileWriter(logs_dir, sess.graph)
    logger.info("Summary created")

    num_train_batches = math.ceil(num / config.BATCH_SIZE)
    MAX_STEP = num_train_batches * end_epoch

    logger.info("Start training!")
    for step in tqdm(range(MAX_STEP)):
        epoch = math.ceil((step+1) / num_train_batches)

        image_batch_array, label_batch_array, bbox_batch_array, landmark_batch_array = sess.run([image_batch,
                                                                                                 label_batch,
                                                                                                 bbox_batch,
                                                                                                 landmark_batch])
        # # for debugging only
        #             # print(type(image_batch_array))
        #             # print(image_batch_array[0].shape)
        #             # print(label_batch_array[0])
        #             # print(bbox_batch_array[0])
        #             # print(landmark_batch_array[0])
        #             # img = Image.fromarray(image_batch_array[0].astype('uint8')).convert('RGB')
        #             # draw = ImageDraw.Draw(img)
        #             # draw.rectangle([(bbox_batch_array[0][0]*image_size, bbox_batch_array[0][1]*image_size),
        #             #                 (bbox_batch_array[0][2]*image_size, bbox_batch_array[0][3]*image_size)],
        #             #                outline='yellow')
        #             # draw.point([(landmark_batch_array[0][2*i]*image_size,
        #             #              landmark_batch_array[0][2*i+1]*image_size) for i in range(5)], fill='red')
        #             # img.show()
        #             # exit()

        #random flip
        image_batch_array, landmark_batch_array = random_flip_images(image_batch_array,
                                                                     label_batch_array,
                                                                     landmark_batch_array)

        _, _, summary = sess.run([train_op, lr_op, summary_op],
                               feed_dict={input_image: image_batch_array,
                                          label: label_batch_array,
                                          bbox_target: bbox_batch_array,
                                          landmark_target: landmark_batch_array})

        if (step+1) % display == 0:
            bbox_loss, landmark_loss, L2_loss = sess.run(
                [bbox_loss_op, landmark_loss_op, L2_loss_op],
                feed_dict={input_image: image_batch_array,
                           label: label_batch_array,
                           bbox_target: bbox_batch_array,
                           landmark_target: landmark_batch_array})

            total_loss = radio_bbox_loss * bbox_loss + \
                         radio_landmark_loss * landmark_loss + \
                         L2_loss
            # landmark loss: %4f,
            logger.info("Step: %d/%d, bbox loss: %4f, Landmark loss :%4f, L2 loss: %4f, Total Loss: %4f"
                  % (step+1,MAX_STEP, bbox_loss,landmark_loss, L2_loss,total_loss))

        if (step+1) % num_train_batches == 0:
            model_output = "mtcnn_epoch{}".format(epoch)
            ckpt_name = os.path.join(model_path, model_output)
            saver.save(sess, ckpt_name)
            logger.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            bbox_loss, landmark_loss, L2_loss, total_loss = eval(data_array=val_data_array,
                                                                 model_path=model_path,
                                                                 img_dir=img_dir,
                                                                 regression=False)
            logger.info("Epoch: %d/%d, bbox loss: %4f, Landmark loss :%4f, L2 loss: %4f, Total Loss: %4f"
                        % (epoch, end_epoch, bbox_loss, landmark_loss, L2_loss, total_loss))
            sess.run(train_init_op)

        writer.add_summary(summary, global_step=step)

    logger.info("Complete!")
    sess.close()
