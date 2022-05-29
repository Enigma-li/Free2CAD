#
# Project Free2CAD
#
#   Author: Changjian Li (chjili2011@gmail.com),
#   Copyright (c) 2022. All Rights Reserved.
#
# ==============================================================================
"""Data loader for Free2CAD.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow as tf
from abc import abstractmethod
from typing import Iterator, Optional
import numpy as np

from libs import decode_aeblock
from libs import decode_gpregblock

# network logger initialization
import logging

loader_logger = logging.getLogger('main.loader')


class BaseReader(object):
    def __init__(self, batch_size, shuffle, raw_size, infinite, name=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.name = name
        self.raw_size = raw_size
        self.infinite = infinite

    @abstractmethod
    def next(self): pass


class AeTFReader(BaseReader):
    def __init__(self, data_dir, batch_size, shuffle, raw_size, infinite, prefix, compress='ZLIB', name=None):
        super().__init__(batch_size, shuffle, raw_size, infinite, name)

        self.record_dir = data_dir
        record_files = [os.path.join(self.record_dir, f) for f in os.listdir(self.record_dir)
                        if f.find(prefix) != -1 and f.endswith('records')]
        loader_logger.info('Load TFRecords: {}'.format(record_files))
        self.raw_size = raw_size
        dataset = tf.data.TFRecordDataset(record_files, compression_type=compress)
        dataset = dataset.map(self.preprocess, tf.data.experimental.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(self.batch_size * 50)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        self.dataset = dataset
        self.iterator: Optional[Iterator] = iter(self.dataset)

    def preprocess(self, example_proto):
        features = tf.io.parse_single_example(example_proto,
                                              features={
                                                  'name': tf.io.FixedLenFeature([], tf.string),
                                                  'block': tf.io.FixedLenFeature([], tf.string)
                                              })

        input_raw = decode_aeblock(features['block'], tensor_size=self.raw_size)
        input_img = tf.reshape(input_raw, [self.raw_size[0], self.raw_size[1], 1])

        return input_img

    def next(self):
        while True:
            try:
                next_elem = next(self.iterator)
                return next_elem
            except StopIteration:
                self.iterator = iter(self.dataset)
                if self.infinite:
                    return self.next()
                else:
                    raise StopIteration


class GPRegTFReader(BaseReader):
    def __init__(self, data_dir, batch_size, shuffle, raw_size, infinite, prefix, compress='ZLIB', name=None):

        super().__init__(batch_size, shuffle, raw_size, infinite, name)

        self.record_dir = data_dir
        record_files = [os.path.join(self.record_dir, f) for f in os.listdir(self.record_dir)
                        if f.find(prefix) != -1 and f.endswith('records')]
        loader_logger.info('Load TFRecords: {}'.format(record_files))
        self.raw_size = raw_size
        self.stroke_padding = -2.0  # do not matter, we will padding again when preparing data
        self.label_padding = -1.0   # same

        dataset = tf.data.TFRecordDataset(record_files, compression_type=compress)
        dataset = dataset.map(self.preprocess, tf.data.experimental.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(self.batch_size * 20)
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=(
                                           tf.TensorShape([raw_size[0], raw_size[1], None]),  # [256, 256, max]
                                           tf.TensorShape([None, None]),  # [max, max]
                                           tf.TensorShape([None, raw_size[0], raw_size[1], 5]),  # [max, 256, 256, 5]
                                           tf.TensorShape([None, 3]),  # [max, 3]
                                           tf.TensorShape([None]),  # [max]
                                           tf.TensorShape([None]),  # [max]
                                           tf.TensorShape([None]),  # [max]
                                           tf.TensorShape([raw_size[0], raw_size[1], 4]),  # global depth, normal maps
                                           tf.TensorShape([None, raw_size[0], raw_size[1], 1]),  # [max, 256, 256, 1], base line segmentation
                                           tf.TensorShape([]),  # scalar, no need to pad
                                           tf.TensorShape([]),  # scalar, no need to pad
                                           tf.TensorShape([raw_size[0], raw_size[1]]),  # [256, 256]
                                           tf.TensorShape([None]),  # [1]
                                           tf.TensorShape([None, raw_size[0], raw_size[1], 1]),   # [max, 256, 256, 1], base face map
                                       ),
                                       padding_values=(self.stroke_padding,
                                                       self.label_padding,
                                                       self.stroke_padding,
                                                       self.stroke_padding,
                                                       self.stroke_padding,
                                                       -2,
                                                       -2,
                                                       self.label_padding,
                                                       self.stroke_padding,
                                                       0,
                                                       0,
                                                       self.stroke_padding,
                                                       0,
                                                       self.stroke_padding),
                                       drop_remainder=True)

        self.dataset = dataset
        self.iterator: Optional[Iterator] = iter(self.dataset)

    def preprocess(self, example_proto):
        features = tf.io.parse_single_example(example_proto,
                                              features={
                                                  'name': tf.io.FixedLenFeature([], tf.string),
                                                  'block': tf.io.FixedLenFeature([], tf.string)
                                              })

        input_raw, glabel_raw, gmap_raw, gdir_raw, gdis_raw, gbstype_raw, gbltype_raw, global_nd, \
        gbseg_raw, preimg_raw, startidx_raw, face_map = decode_gpregblock(features['block'], tensor_size=self.raw_size)
        
        nb_gp = tf.shape(glabel_raw)[0]
        nb_stroke = tf.shape(glabel_raw)[1]

        return input_raw, glabel_raw, gmap_raw, gdir_raw, gdis_raw, gbstype_raw, gbltype_raw, global_nd, \
                gbseg_raw, nb_gp, nb_stroke, preimg_raw, startidx_raw, face_map

    def next(self):
        while True:
            try:
                next_elem = next(self.iterator)
                return next_elem
            except StopIteration:
                self.iterator = iter(self.dataset)
                if self.infinite:
                    return self.next()
                else:
                    raise StopIteration
