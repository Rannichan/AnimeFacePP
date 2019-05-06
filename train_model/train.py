"""
yipengwa@usc.edu
"""
import os
import math
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
from train_config import config
from data_loader import batch_generator, load_all
from train_utils import train_model, image_color_distort
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


def train(net_factory, model_path, img_dir,
          label_files, val_label_files,
          end_epoch=20, display=100, base_lr=0.01):
    """
    train PNet/RNet/ONet
    :param net_factory: pre-defined model structure
    :param model_path: str, model path
    :param img_dir: directory for saving images and label files
    :param label_files: list of paths of label files
    :param val_label_files: list of paths of validation label files
    :param end_epoch: int, total training epoch
    :param display: every $display$ steps show the training loss
    :param base_lr: initial learning rate
    :return:
    """
    net = "Manga_Net"
    image_size = 48
    if not os.path.exists(model_path): os.makedirs(model_path)
    logger.info("Model will be saved in {}".format(model_path))
    num = 0
    for label_file in label_files:
        with open(label_file, 'r') as f:
            logger.info("Label file found: {}".format(label_file))
            num += len(f.readlines())
    logger.info("Size of the dataset is: {}".format(num))

    # data loader for training
    train_batches = batch_generator(img_dir, label_files, config.BATCH_SIZE, image_size)
    iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
    image_batch, label_batch, bbox_batch = iter.get_next()
    train_init_op = iter.make_initializer(train_batches)
    logger.info("Data loader created")

    # load data for validation
    _, val_image_array, val_label_array, val_bbox_array = load_all(img_dir, val_label_files, image_size)

    radio_cls_loss = 0.5
    radio_bbox_loss = 0.5

    # define placeholder
    input_image = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.int32, shape=[None], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[None, 4], name='bbox_target')
    logger.info("Input tensor placeholder defined")

    # get loss and accuracy
    input_image = image_color_distort(input_image)
    cls_loss_op, bbox_loss_op, L2_loss_op, _, _ = net_factory(inputs=input_image,
                                                        label=label,
                                                        bbox_target=bbox_target)
    total_loss_op = radio_cls_loss * cls_loss_op + radio_bbox_loss * bbox_loss_op + L2_loss_op
    train_op, lr_op = train_model(base_lr, total_loss_op, num)
    logger.info("Loss operation and training operation defined")

    # # for debugging
    # for var in tf.all_variables():
    #     print(var)
    # exit()

    saver = tf.train.Saver(max_to_keep=0)
    sess = tf.Session()
    ckpt = tf.train.latest_checkpoint(model_path)
    if ckpt is None:
        logger.info("Initialize from scratch")
        sess.run(tf.global_variables_initializer())
    else:
        logger.info("Initialize from exiting model")
        saver.restore(sess, ckpt)
    sess.run(train_init_op)
    logger.info("Graph initialized")

    # visualize some variables
    tf.summary.scalar("bbox_loss", bbox_loss_op)
    tf.summary.scalar("landmark_loss", cls_loss_op)
    tf.summary.scalar("total_loss", total_loss_op)
    summary_op = tf.summary.merge_all()
    logs_dir = "../logs/{}".format(net)
    if not os.path.exists(logs_dir): os.makedirs(logs_dir)
    writer = tf.summary.FileWriter(logs_dir, sess.graph)
    logger.info("Summary created")

    num_train_batches = math.ceil(num / config.BATCH_SIZE)
    MAX_STEP = num_train_batches * end_epoch
    graph = tf.get_default_graph()
    global_step = graph.get_tensor_by_name('global_step:0')
    gs_ = sess.run(global_step)

    logger.info("Start training!")
    for step in tqdm(range(gs_, MAX_STEP)):
        epoch = math.ceil((step + 1) / num_train_batches)

        image_batch_array, label_batch_array, bbox_batch_array = sess.run([image_batch, label_batch, bbox_batch])

        # # for debugging only
        # print(type(image_batch_array))
        # print(image_batch_array[0].shape)
        # print(label_batch_array[0])
        # print(bbox_batch_array[0])
        # print(landmark_batch_array[0])
        # img = Image.fromarray(image_batch_array[0].astype('uint8')).convert('RGB')
        # draw = ImageDraw.Draw(img)
        # draw.rectangle([(bbox_batch_array[0][0]*image_size, bbox_batch_array[0][1]*image_size),
        #                 (bbox_batch_array[0][2]*image_size, bbox_batch_array[0][3]*image_size)],
        #                outline='yellow')
        # draw.point([(landmark_batch_array[0][2*i]*image_size,
        #              landmark_batch_array[0][2*i+1]*image_size) for i in range(5)], fill='red')
        # img.show()
        # exit()

        _, _, summary = sess.run([train_op, lr_op, summary_op],
                                 feed_dict={input_image: image_batch_array,
                                            label: label_batch_array,
                                            bbox_target: bbox_batch_array})

        if (step + 1) % display == 0:
            cls_loss, bbox_loss, L2_loss = sess.run(
                [cls_loss_op, bbox_loss_op, L2_loss_op],
                feed_dict={input_image: image_batch_array,
                           label: label_batch_array,
                           bbox_target: bbox_batch_array})
            total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * bbox_loss + L2_loss
            logger.info("Step: %d/%d, cls loss : %4f, bbox loss: %4f, L2 loss: %4f, total Loss: %4f"
                        % (step + 1, MAX_STEP, cls_loss, bbox_loss, L2_loss, total_loss))

        if (step + 1) % num_train_batches == 0:
            model_output = "manga_net_epoch{}".format(epoch)
            ckpt_name = os.path.join(model_path, model_output)
            saver.save(sess, ckpt_name)
            logger.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            cls_loss, bbox_loss, L2_loss = sess.run(
                [cls_loss_op, bbox_loss_op, L2_loss_op],
                feed_dict={input_image: val_image_array,
                           label: val_label_array,
                           bbox_target: val_bbox_array})
            total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * bbox_loss + L2_loss
            logger.info("Epoch: %d/%d, cls loss: %4f, bbox loss: %4f, L2 loss: %4f, Total Loss: %4f"
                        % (epoch, end_epoch, cls_loss, bbox_loss, L2_loss, total_loss))

            sess.run(train_init_op)

        writer.add_summary(summary, global_step=step)
    logger.info("Complete!")
    sess.close()
