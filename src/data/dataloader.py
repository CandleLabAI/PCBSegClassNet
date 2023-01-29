# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from glob import glob

# class_mapping is used to encode label into one hot mapping.
class_mapping = {
    "R": 0,
    "C": 1,
    "U": 2,
    "Q": 3,
    "J": 4,
    "L": 5,
    "RA": 6,
    "D": 7,
    "RN": 8,
    "TP": 9,
    "IC": 10,
    "P": 11,
    "CR": 12,
    "M": 13,
    "BTN": 14,
    "FB": 15,
    "CRA": 16,
    "SW": 17,
    "T": 18,
    "F": 19,
    "V": 20,
    "LED": 21,
    "S": 22,
    "QA": 23,
    "JP": 24,
}

# color_values is used to encode mask into one hot mapping. following color needs to be updated based on the color used while creating masks
color_values = {
    0: (0, 0, 30),
    1: (0, 0, 60),
    2: (0, 0, 90),
    3: (0, 0, 120),
    4: (0, 0, 150),
    5: (0, 0, 180),
    6: (0, 0, 210),
    7: (0, 0, 240),
    8: (0, 30, 30),
    9: (0, 30, 60),
    10: (0, 30, 90),
    11: (0, 30, 120),
    12: (0, 30, 150),
    13: (0, 30, 180),
    14: (0, 30, 210),
    15: (0, 30, 240),
    16: (0, 60, 30),
    17: (0, 60, 60),
    18: (0, 60, 90),
    19: (0, 60, 120),
    20: (0, 60, 150),
    21: (0, 60, 180),
    22: (0, 60, 210),
    23: (0, 60, 240),
    24: (0, 90, 30),
    25: (0, 0, 0),
}


def get_paths(opt):
    if opt["type"] == "Segmentation":
        images = sorted(
            [
                os.path.join(opt["data_images"], img)
                for img in os.listdir(opt["data_images"])
            ]
        )
        masks = sorted(
            [
                os.path.join(opt["data_masks"], msk)
                for msk in os.listdir(opt["data_masks"])
            ]
        )
        return images, masks
    elif opt["type"] == "Classification":
        images = glob(os.path.join(opt["data_images"], "*/*"))
        labels = [class_mapping[str(lbl.split(os.path.sep)[-2])] for lbl in images]
        return images, labels
    else:
        assert (
            False
        ), f"Found data type as {opt['type']} but it should be one of Segmentation/Classification"


class LoadSegData:
    def __init__(self, opt):
        self.opt = opt

    def parse_data(self, image, mask):
        # read the image from disk, decode it, convert the data type to
        # floating point, and resize it
        image = tf.io.read_file(image)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize(image, (self.opt["img_size_h"], self.opt["img_size_w"]))

        mask = tf.io.read_file(mask)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.image.resize(mask, (self.opt["img_size_h"], self.opt["img_size_w"]))

        one_hot_map = []
        for colour in list(color_values.values()):
            class_map = tf.reduce_all(tf.equal(mask, colour), axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)

        # return the image and the label
        return image, one_hot_map

    def init(self, images, masks):
        ds = tf.data.Dataset.from_tensor_slices((images, masks))
        if self.opt["use_shuffle"]:
            ds = (
                ds.shuffle(len(images))
                .map(self.parse_data, num_parallel_calls=AUTOTUNE)
                .batch(self.opt["batch_size"], drop_remainder=True)
                .prefetch(AUTOTUNE)
            )
        else:
            ds = (
                ds.map(self.parse_data, num_parallel_calls=AUTOTUNE)
                .batch(self.opt["batch_size"], drop_remainder=True)
                .prefetch(AUTOTUNE)
            )
        return ds


class LoadClassData:
    def __init__(self, opt, num_classes):
        self.opt = opt
        self.num_classes = num_classes

    def parse_data(self, image, label):
        # read the image from disk, decode it, convert the data type to
        # floating point, and resize it
        image = tf.io.read_file(image)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize(image, (self.opt["img_size_h"], self.opt["img_size_w"]))

        label = label

        # return the image and the label
        return image, label

    def init(self, images, labels):
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if self.opt["use_shuffle"]:
            ds = (
                ds.shuffle(len(images))
                .map(self.parse_data, num_parallel_calls=AUTOTUNE)
                .batch(self.opt["batch_size"], drop_remainder=True)
                .prefetch(AUTOTUNE)
            )
        else:
            ds = (
                ds.map(self.parse_data, num_parallel_calls=AUTOTUNE)
                .batch(self.opt["batch_size"], drop_remainder=True)
                .prefetch(AUTOTUNE)
            )
        return ds


def get_data(opt):
    train_opt = opt["datasets"]["train"]
    val_opt = opt["datasets"]["val"]
    num_classes = opt["train"]["num_classes"]

    # train
    train_images, train_targets = get_paths(train_opt)
    val_images, val_targets = get_paths(val_opt)

    if train_opt["type"] == "Segmentation":
        train_data = LoadSegData(train_opt)
        trainDS = train_data.init(train_images, train_targets)

    elif train_opt["type"] == "Classification":
        train_data = LoadClassData(train_opt, num_classes)
        trainDS = train_data.init(train_images, train_targets)

    else:
        assert (
            False
        ), f"Found data type as {train_opt['type']} but it should be one of Segmentation/Classification"

    if val_opt["type"] == "Segmentation":
        val_data = LoadSegData(val_opt)
        valDS = val_data.init(val_images, val_targets)

    elif val_opt["type"] == "Classification":
        val_data = LoadClassData(val_opt, num_classes)
        valDS = val_data.init(val_images, val_targets)

    else:
        assert (
            False
        ), f"Found data type as {val_opt['type']} but it should be one of Segmentation/Classification"

    return trainDS, valDS
