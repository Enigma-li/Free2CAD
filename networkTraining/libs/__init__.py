import tensorflow as tf
import os

_current_path = os.path.dirname(os.path.realpath(__file__))
_tf_decoder_module = tf.load_op_library(os.path.join(_current_path, 'custom_decoder.so'))
decode_aeblock = _tf_decoder_module.decode_aeblock
decode_gpregblock = _tf_decoder_module.decode_gpregblock
