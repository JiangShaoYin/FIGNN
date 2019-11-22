import tensorflow as tf
import numpy as np

a=np.arange(6).reshape(2, 3)
b = tf.convert_to_tensor(a)

c = np.arange(6).reshape(2, 3)
d = tf.convert_to_tensor(c)


with tf.Session() as sess:

    print(sess.run(b+d))










