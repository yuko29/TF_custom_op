import tensorflow as tf 
from tensorflow import keras
import tensorflow_zero_out 
import numpy as np
import os

from tensorflow.python.framework import graph_util


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# @tf.function
# def test(x):
#     return tensorflow_zero_out.zero_out(x)

# y = tf.random.uniform([2], maxval=10, dtype=tf.int32)

# print(test(y))

pb_file_path = os.getcwd() + '/save_model/'

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    x = tf.compat.v1.placeholder(tf.int32, name='x')
    y = tf.compat.v1.placeholder(tf.int32, name='y')
    b = tf.Variable(1, name='b')
    xy = tf.multiply(x, y)
    z = tensorflow_zero_out.zero_out(xy)
    # 这里的输出需要加上name属性
    op = tf.add(z, b, name='op_to_store')

    sess.run(tf.compat.v1.global_variables_initializer())

    # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

    # 测试 OP
    feed_dict = {x: 10, y: 3}
    print(sess.run(op, feed_dict))

    # 写入序列化的 PB 文件
    with tf.compat.v1.gfile.FastGFile(pb_file_path+'model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    # 输出

