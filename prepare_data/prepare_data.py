"""
yipengwa@usc.edu
"""
import os
import sys
import cv2
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw
from bbox import BBox
from prep_utils import IoU
from landmark_utils import rotate, flip


def gen_img_bbox(img_dir, label_file, net):
    """
    gen image and corresponding label file for bbox regression task training
    :param img_dir:
    :param label_file:
    :param net:
    :return:
    """
    if net == "PNet": img_size = 12
    elif net == "RNet": img_size = 24
    elif net == "ONet": img_size = 48
    else:
        print('Net type error')
        return

    img_idx = 0
    pos_label_file = open(os.path.join(img_dir, 'pos_{}.txt'.format(img_size)), 'w')
    part_label_file = open(os.path.join(img_dir, 'part_{}.txt'.format(img_size)), 'w')
    pos_dir = os.path.join(img_dir, 'pos_{}'.format(img_size))
    part_dir = os.path.join(img_dir, 'part_{}'.format(img_size))
    if not os.path.exists(pos_dir): os.makedirs(pos_dir)
    if not os.path.exists(part_dir): os.makedirs(part_dir)

    label_file = read_label_file2(label_file)
    img_idx = 0
    for data_unit in label_file:
        filename = data_unit["filename"]
        bbox = data_unit["bbox"]

        # read image
        img_path = os.path.join(img_dir, filename).replace('\\','/')
        img = cv2.imread(img_path)
        height, width, channel = img.shape

        # read bounding box
        xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
        box = [xmin, ymin, xmax, ymax]
        box = np.array([float(int(x)) for x in box])
        box_width = xmax - xmin + 1
        box_height = ymax - ymin + 1

        # ignore too small faces or incomplete faces
        if max(box_width, box_height) < 20 or xmin < 0 or ymin < 0:
            continue

        for i in range(10):
            # pos and part face size [minsize*0.8,maxsize*1.25]
            size = np.random.randint(int(min(box_width, box_height) * 0.8),
                                     np.ceil(max(box_width, box_height) * 1.25))

            # offset of box center
            delta_x = np.random.randint(int(-box_width * 0.2), int(box_width * 0.2))
            delta_y = np.random.randint(int(-box_height * 0.2), int(box_height * 0.2))

            new_xmin = int(max(xmin + box_width / 2 + delta_x - size / 2, 0))
            new_ymin = int(max(ymin + box_height / 2 + delta_y - size / 2, 0))
            new_xmax = new_xmin + size
            new_ymax = new_ymin + size
            # discard wrong new box
            if new_xmax > width or new_ymax > height:
                continue
            else:
                crop_box = np.array([new_xmin, new_ymin, new_xmax, new_ymax])

            # get normalized bounding box
            offset_x1 = (xmin - new_xmin) / float(size)
            offset_y1 = (ymin - new_ymin) / float(size)
            offset_x2 = (xmax - new_xmin) / float(size)
            offset_y2 = (ymax - new_ymin) / float(size)

            # crop new box form original image, and resize into the preset size
            cropped_im = img[new_ymin:new_ymax, new_xmin:new_xmax, :]
            resized_im = cv2.resize(cropped_im, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            iou = IoU(crop_box, box_)
            if iou >= 0.65:
                save_file = os.path.join(pos_dir, "{}.jpg".format(img_idx))
                cv2.imwrite(save_file, resized_im)
                pos_label_file.write("pos_{}/{}.jpg".format(img_size, img_idx) +
                                     ' 1 {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(offset_x1, offset_y1,
                                                                               offset_x2, offset_y2))
                img_idx += 1
            elif iou >= 0.4:
                save_file = os.path.join(part_dir, "{}.jpg".format(img_idx))
                cv2.imwrite(save_file, resized_im)
                part_label_file.write("part_{}/{}.jpg".format(img_size, img_idx) +
                                     ' -1 {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(offset_x1, offset_y1,
                                                                                offset_x2, offset_y2))
                img_idx += 1
            sys.stdout.write('\r>> {} images has been generated'.format(img_idx))
            sys.stdout.flush()


def gen_img_landmark(img_dir, label_file, net, argument=False):
    """
    gen image and corresponding label file for landmarks regression task training
    :param img_dir:
    :param label_file:
    :param net:
    :param argument:
    :return:
    """
    if net == "PNet": size = 12
    elif net == "RNet": size = 24
    elif net == "ONet": size = 48
    else:
        print('Net type error')
        return

    img_idx = 0
    total = 0
    landmark_dir = os.path.join(img_dir, 'landmark_{}'.format(size))
    if not os.path.exists(landmark_dir): os.makedirs(landmark_dir)
    landmark_label_file = open(os.path.join(img_dir, "landmark_{}.txt".format(size)), 'w')
    label_file = read_label_file2(label_file)
    for data_unit in label_file:
        filename = data_unit["filename"]
        bbox = data_unit["bbox"]

        # read image
        img_path = os.path.join(img_dir, filename).replace('\\','/')
        img = cv2.imread(img_path)
        height, width, channel = img.shape

        # read bounding box
        xmin, ymin, xmax, ymax = int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])
        box = [xmin, ymin, xmax, ymax]
        box = np.array([float(x) for x in box])
        box_width = xmax - xmin + 1
        box_height = ymax - ymin + 1

        # read landmarks
        landmark1 = [bbox["xlefteye"], bbox["ylefteye"]]
        landmark2 = [bbox["xrighteye"], bbox["yrighteye"]]
        landmark3 = [bbox["xnose"], bbox["ynose"]]
        landmark4 = [bbox["xleftmouth"], bbox["yleftmouth"]]
        landmark5 = [bbox["xrightmouth"], bbox["yrightmouth"]]
        landmark = np.concatenate((landmark1, landmark2, landmark3, landmark4, landmark5), axis=0)
        landmark1_normalized = [(bbox["xlefteye"]-xmin)/box_width, (bbox["ylefteye"]-ymin)/box_height]
        landmark2_normalized = [(bbox["xrighteye"]-xmin)/box_width, (bbox["yrighteye"]-ymin)/box_height]
        landmark3_normalized = [(bbox["xnose"]-xmin)/box_width, (bbox["ynose"]-ymin)/box_height]
        landmark4_normalized = [(bbox["xleftmouth"]-xmin)/box_width, (bbox["yleftmouth"]-ymin)/box_height]
        landmark5_normalized = [(bbox["xrightmouth"]-xmin)/box_width, (bbox["yrightmouth"]-ymin)/box_height]
        landmark_normalized = np.concatenate((landmark1_normalized,
                                              landmark2_normalized,
                                              landmark3_normalized,
                                              landmark4_normalized,
                                              landmark5_normalized), axis=0)


        # crop face bounding box form original image, and resize into the preset size
        cropped_im = img[ymin:ymax, xmin:xmax, :]
        resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)

        save_file = os.path.join(landmark_dir, "{}.jpg".format(img_idx))
        cv2.imwrite(save_file, resized_im)
        landmark_label_file.write("landmark_{}/{}.jpg".format(size, img_idx) + " -2 " +
                                  " ".join([str(x) for x in landmark_normalized]) + "\n")
        img_idx += 1
        total += 1

        img_aug = []
        landmark_aug = []
        if argument:
            new_landmark = np.zeros((5,2))

            # ignore too small faces or incomplete faces
            if max(box_width, box_height) < 40 or xmin < 0 or ymin < 0:
                continue

            # random shift the bounding box to get augmented data
            for i in range(5):
                bbox_size = np.random.randint(int(min(box_width, box_height) * 0.8),
                                              np.ceil(max(box_width, box_height) * 1.25))
                delta_x = np.random.randint(int(-box_width * 0.2), int(box_width * 0.2))
                delta_y = np.random.randint(int(-box_height * 0.2), int(box_height * 0.2))

                new_xmin = int(max(xmin + box_width / 2 + delta_x - bbox_size / 2, 0))
                new_ymin = int(max(ymin + box_height / 2 + delta_y - bbox_size / 2, 0))
                new_xmax = new_xmin + bbox_size
                new_ymax = new_ymin + bbox_size

                # discard wrong new box
                if new_xmax > width or new_ymax > height:
                    continue
                else:
                    crop_box = np.array([new_xmin, new_ymin, new_xmax, new_ymax])

                # crop new box form original image, and resize into the preset size
                cropped_im = img[new_ymin:new_ymax+1, new_xmin:new_xmax+1, :]
                resized_im = cv2.resize(cropped_im, (size, size))

                iou = IoU(crop_box, np.expand_dims(box, 0))
                if iou > 0.65:
                    # augmentation 1
                    img_aug.append(resized_im)
                    # normalize
                    for index, landmark_i in enumerate(landmark.reshape((5,2))):
                        rv = ((landmark_i[0] - new_xmin) / bbox_size,
                              (landmark_i[1] - new_ymin) / bbox_size)
                        new_landmark[index] = rv
                    landmark_aug.append(new_landmark.reshape(10))

                    bbox = BBox([new_xmin, new_ymin, new_xmax, new_ymax])

                    # augmentation 2 -- mirror
                    if random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, new_landmark)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        img_aug.append(face_flipped)
                        landmark_aug.append(landmark_flipped.reshape(10))

                    # augmentation 3 -- clockwise rotation
                    if random.choice([0, 1]) > 0:
                        face_rotated, landmark_rotated = rotate(img, bbox,
                                                                bbox.reprojectLandmark(new_landmark), 5)
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated = cv2.resize(face_rotated, (size, size))
                        img_aug.append(face_rotated)
                        landmark_aug.append(landmark_rotated.reshape(10))

                        # flip
                        face_flipped, landmark_flipped = flip(face_rotated, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        img_aug.append(face_flipped)
                        landmark_aug.append(landmark_flipped.reshape(10))

                    # augmentation 4 -- anti-clockwise rotation
                    if random.choice([0, 1]) > 0:
                        face_rotated, landmark_rotated = rotate(img, bbox,
                                                                bbox.reprojectLandmark(new_landmark), -5)
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated = cv2.resize(face_rotated, (size, size))
                        img_aug.append(face_rotated)
                        landmark_aug.append(landmark_rotated.reshape(10))

                        face_flipped, landmark_flipped = flip(face_rotated, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        img_aug.append(face_flipped)
                        landmark_aug.append(landmark_flipped.reshape(10))

            for i in range(len(img_aug)):
                if min(landmark_aug[i]) <= 0  or max(landmark_aug[i]) >= 1:
                    continue
                save_file = os.path.join(landmark_dir, "{}_{}.jpg".format(img_idx, i))
                cv2.imwrite(save_file, img_aug[i])
                landmark_label_file.write("landmark_{}/{}_{}.jpg".format(size, img_idx, i) + " -2 " +
                                          " ".join([str(x) for x in landmark_aug[i]]) + "\n")
                total += 1

        sys.stdout.write('\r>> {} images has been generated'.format(total))
        sys.stdout.flush()


def gen_img_test(img_dir, test_label_file, net):
    """
    generate image and corresponding label file for test or evaluation
    :param img_dir:
    :param test_label_file:
    :param net:
    :return:
    """
    if net == "PNet": size = 12
    elif net == "RNet": size = 24
    elif net == "ONet": size = 48
    else:
        print('Net type error')
        return

    img_idx = 0

    bbox_label_file = open(os.path.join(img_dir, 'test_bbox_{}.txt'.format(size)), 'w')
    landmark_label_file = open(os.path.join(img_dir, "test_landmark_{}.txt".format(size)), 'w')
    test_dir = os.path.join(img_dir, 'test_{}'.format(size))  # two label files share one image directory
    if not os.path.exists(test_dir): os.makedirs(test_dir)

    label_file = read_label_file2(test_label_file)
    for data_unit in label_file:
        filename = data_unit["filename"]
        bbox = data_unit["bbox"]

        # read image
        img_path = os.path.join(img_dir, filename).replace('\\', '/')
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        if height != width: continue

        # read normalized bounding box
        xmin, ymin, xmax, ymax = int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])
        box_normalized = [float(xmin)/width, float(ymin)/height, float(xmax)/width, float(ymax)/height]

        # read normalized landmarks
        landmark1 = [bbox["xlefteye"]/width, bbox["ylefteye"]/height]
        landmark2 = [bbox["xrighteye"]/width, bbox["yrighteye"]/height]
        landmark3 = [bbox["xnose"]/width, bbox["ynose"]/height]
        landmark4 = [bbox["xleftmouth"]/width, bbox["yleftmouth"]/height]
        landmark5 = [bbox["xrightmouth"]/width, bbox["yrightmouth"]/height]
        landmark_normalized = np.concatenate((landmark1, landmark2, landmark3, landmark4, landmark5), axis=0)

        resized_im = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        save_file = os.path.join(test_dir, "{}.jpg".format(img_idx))
        cv2.imwrite(save_file, resized_im)
        bbox_label_file.write("test_{}/{}.jpg".format(size, img_idx) + " -1 " +
                              " ".join([str(x) for x in box_normalized]) + "\n")
        landmark_label_file.write("test_{}/{}.jpg".format(size, img_idx) + " -2 " +
                                  " ".join([str(x) for x in landmark_normalized]) + "\n")
        img_idx += 1
        sys.stdout.write('\r>> {} images has been generated'.format(img_idx))
        sys.stdout.flush()


def read_label_file(label_file, with_landmark=True):
    """
    read data from given label file
    :param img_dir: str, directory shared by label files and images
    :param label_file: str, absolute path of label file
    :param with_landmark:
    :return: List of 3-element-tuple, (img_path, bbox_tuple, landmark_tuple)
    """
    result = []

    with open(label_file, 'r') as lf:
        for line in lf:
            data_units = line.strip().split()
            # read absolute path of image
            img_path = data_units[0].replace('\\','/')
            # read bounding box (x1, y1, x2, y2)
            bbox = [data_units[1], data_units[3], data_units[2], data_units[4]]
            bbox = [int(float(x)) for x in bbox]
            # read landmarks (x1, )
            if with_landmark:
                landmarks = np.zeros((5,2))
                for i in range(5):
                    landmarks[i] = (float(data_units[5+2*i]),
                                    float(data_units[6+2*i]))
                result.append((img_path, BBox(bbox), landmarks))

    return result


def read_label_file2(label_file):
    print('Read label file 2: the label file is ', label_file)
    imagelist = open(label_file, 'r')
    dataset = []
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
        dataset.append(data_unit)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test", help="train_bbox/train_landmark/test", type=str)
    parser.add_argument("--net", default="RNet", help="model name", type=str)
    parser.add_argument("--img_dir", default="../data/bbx_landmark", help="image directory", type=str)
    parser.add_argument("--label_file", default="../data/bbx_landmark/testImageList.txt", help="original label file", type=str)
    args = parser.parse_args()
    mode = args.mode
    net = args.net
    img_dir = args.img_dir
    label_file = args.label_file
    # net = 'RNet'
    # img_dir = '../data/bbx_landmark'
    # train_label_file = '../data/bbx_landmark/trainImageList.txt'
    # test_label_file = '../data/bbx_landmark/testImageList.txt'
    # output_dir = '../data/train/RNet'

    if mode == "test":
        gen_img_test(img_dir, label_file, net)
    elif mode == "train_bbox":
        gen_img_bbox(img_dir, label_file, net)
    elif mode == "train_landmark":
        gen_img_landmark(img_dir, label_file, net, argument=True)
    else:
        print("Mode error, pick one from train_bbox/train_landmark/test")
