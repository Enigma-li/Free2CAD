TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared decode_aeblock_op.cc decode_gpregblock_op.cc -o custom_decoder.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
