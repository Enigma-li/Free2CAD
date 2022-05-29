#
# Project Free2CAD
#
#   Author: Changjian Li (chjili2011@gmail.com),
#   Copyright (c) 2022. All Rights Reserved.
#
# ==============================================================================
"""Convert LMDB to TFRecords
"""

import lmdb
import tensorflow as tf
import os

tf_fn = r'\path\to\tfrecord'
data_dir = r'\path\to\lmdb'


def __bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def lmdb_to_TFRecords():
    options_zlib = tf.io.TFRecordOptions(compression_type='ZLIB')
    tf_writer = tf.io.TFRecordWriter(tf_fn, options=options_zlib)

    # collect all lmdbs to write into one TFRecords (at least one lmdb)
    db_paths = [data_dir]
    
    cnt = 0
    for i in range(1):
        env = lmdb.open(db_paths[i], readonly=True)
        with env.begin() as txn:
            with txn.cursor() as curs:

                for key, value in curs:
                    feature = {
                        'name': __bytes_feature(key),
                        'block': __bytes_feature(value)
                    }
                    
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    
                    tf_writer.write(example.SerializeToString())
                    print('put key: {} to tfrecord'.format(key.decode('utf-8')))
                    
                    cnt += 1
                    
                    if cnt > 9999:
                      break

    tf_writer.close()
    
    print('Summary: total {}'.format(cnt))


if __name__ == '__main__':
    # Set GPU (could remove this setting when running on machine without GPU)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    lmdb_to_TFRecords()
