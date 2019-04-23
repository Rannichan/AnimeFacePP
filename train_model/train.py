"""
yipengwa@usc.edu
"""
import os
import sys
sys.path.append("../prepare_data")
sys.path.append("../train_model")
import math
import random
import cv2
from datetime import datetime
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
from train_model.MTCNN_config import config
from data_loader import batch_generator
from PIL import ImageDraw, Image


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
    #LR_EPOCH [8,14]
    #boundaried [num_batch,num_batch]
    num_train_batches = math.ceil(data_num / config.BATCH_SIZE)
    boundaries = [int(epoch * num_train_batches) for epoch in config.LR_EPOCH]
    #lr_values[0.01, 0.001, 0.0001, 0.00001]
    lr_values = [base_lr, base_lr * warmup_factor] + [base_lr * (lr_factor ** x) for x in range(1, len(config.LR_EPOCH))]
    #control learning rate
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)
    return train_op, lr_op


# all mini-batch mirror
def random_flip_images(image_batch,label_batch,landmark_batch):
    #mirror
    if random.choice([0,1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch==-2)[0]
        flipposindexes = np.where(label_batch==1)[0]
        #only flip
        flipindexes = np.concatenate((fliplandmarkindexes,flipposindexes))
        #random flip    
        for i in flipindexes:
            cv2.flip(image_batch[i],1,image_batch[i])        
        
        #pay attention: flip landmark    
        for i in fliplandmarkindexes:
            landmark_ = landmark_batch[i].reshape((-1,2))
            landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
            landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth        
            landmark_batch[i] = landmark_.ravel()
        
    return image_batch,landmark_batch


def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)
    return inputs


def train(net_factory, model_path, img_dir, label_files,
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
    net = model_path.split('/')[-1]
    print("Current training net: {}".format(net))
    print("Model will be saved in {}".format(model_path))
    # label_file = '../data/bbx_landmark/label_24_aug.txt'
    num = 0
    for label_file in label_files:
        with open(label_file, 'r') as f:
            print("Label file found: {}".format(label_file))
            num += len(f.readlines())
    print("Size of the dataset is: ", num)

    # data loader for training
    train_batches = batch_generator(img_dir, label_files, config.BATCH_SIZE, net)
    iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
    image_batch, label_batch, bbox_batch, landmark_batch = iter.get_next()
    train_init_op = iter.make_initializer(train_batches)
    print("------------Data loader created------------")
        
    #landmark_dir    
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
    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32,shape=[config.BATCH_SIZE,10],name='landmark_target')
    print("------------Input tensor placeholder defined------------")

    # get loss and accuracy
    input_image = image_color_distort(input_image)
    bbox_loss_op, landmark_loss_op, L2_loss_op = net_factory(inputs=input_image,
                                                             label=label,
                                                             bbox_target=bbox_target,
                                                             landmark_target=landmark_target,
                                                             training=True)
    total_loss_op  = radio_bbox_loss * bbox_loss_op + \
                     radio_landmark_loss * landmark_loss_op + \
                     L2_loss_op
    train_op, lr_op = train_model(base_lr, total_loss_op, num)
    print("------------Loss operation and training operation defined------------")

    # # for debugging
    # for var in tf.all_variables():
    #     print(var)
    # exit()

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=4)
    sess.run(train_init_op)
    # sess.run(val_init_op)
    sess.run(init)
    print("------------Graph initialized------------")

    #visualize some variables
    tf.summary.scalar("bbox_loss", bbox_loss_op)
    tf.summary.scalar("landmark_loss", landmark_loss_op)
    tf.summary.scalar("total_loss", total_loss_op)
    summary_op = tf.summary.merge_all()
    logs_dir = "../logs/{}".format(net)
    if os.path.exists(logs_dir) == False:
        os.makedirs(logs_dir)
    writer = tf.summary.FileWriter(logs_dir, sess.graph)
    print("------------Summary created------------")

    num_train_batches = math.ceil(num / config.BATCH_SIZE)
    MAX_STEP = num_train_batches * end_epoch
    try:
        print("------------Start training!------------")
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
                print("%s : Step: %d/%d, bbox loss: %4f, Landmark loss :%4f, L2 loss: %4f, Total Loss: %4f"
                      % (datetime.now(), step+1,MAX_STEP, bbox_loss,landmark_loss, L2_loss,total_loss))

            if (step+1) % num_train_batches == 0:
                model_output = "mtcnn_epoch{}".format(epoch)
                ckpt_name = os.path.join(model_path, model_output)
                saver.save(sess, model_path, global_step=step)
                print("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))
                sess.run(train_init_op)

            writer.add_summary(summary, global_step=step)

    except tf.errors.OutOfRangeError:
        print("Complete!")

    sess.close()
