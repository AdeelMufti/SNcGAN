import numpy as np
from math import ceil

from libs.ops import *

def conv_out_size_same(size, stride):
  return int(ceil(float(size) / float(stride)))

class DCGANGenerator(object):
  def __init__(self, hidden_dim=128, batch_size=64, hidden_activation=tf.nn.relu, output_activation=tf.nn.tanh, use_batch_norm=True, z_distribution='normal', scope='generator', image_size_height=128, image_size_width=128, stride=2, dropout=None, **kwargs):
    self.hidden_dim = hidden_dim
    self.batch_size = batch_size
    self.hidden_activation = hidden_activation
    self.output_activation = output_activation
    self.use_batch_norm = use_batch_norm
    self.z_distribution = z_distribution
    self.scope = scope
    self.image_size_height = image_size_height
    self.image_size_width = image_size_width
    self.stride = stride
    self.dropout = dropout

  def __call__(self, z, y=None, stride=2, is_training=True, **kwargs):
    with tf.variable_scope(self.scope):
      s_h, s_w = self.image_size_height, self.image_size_width
      s2_h, s2_w = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s4_h, s4_w = conv_out_size_same(s2_h, 2), conv_out_size_same(s2_w, 2)
      s8_h, s8_w = conv_out_size_same(s4_h, 2), conv_out_size_same(s4_w, 2)

      if self.use_batch_norm:
        if y is not None:
          inputs = tf.concat(axis=1, values=[z, y])
        else:
          inputs = z

        if self.dropout:
          inputs = tf.layers.dropout(inputs, rate=self.dropout)

        l0  = self.hidden_activation(batch_norm(linear(inputs, s8_h * s8_w * 512, name='l0', stddev=0.02), name='bn0', is_training=is_training))
        l0  = tf.reshape(l0, [self.batch_size, s8_h, s8_w, 512])
        if self.dropout:
          l0 = tf.layers.dropout(l0, rate=self.dropout)

        dc1 = self.hidden_activation(batch_norm(deconv2d( l0, [self.batch_size,  s4_h, s4_w, 256], name='dc1', stddev=0.02), name='bn1', is_training=is_training))
        if self.dropout:
          dc1 = tf.layers.dropout(dc1, rate=self.dropout)

        dc2 = self.hidden_activation(batch_norm(deconv2d(dc1, [self.batch_size, s2_h, s2_w, 128], name='dc2', stddev=0.02), name='bn2', is_training=is_training))
        if self.dropout:
          dc2 = tf.layers.dropout(dc2, rate=self.dropout)

        dc3 = self.hidden_activation(batch_norm(deconv2d(dc2, [self.batch_size, s_h, s_w,  64], name='dc3', stddev=0.02), name='bn3', is_training=is_training))
        if self.dropout:
          dc3 = tf.layers.dropout(dc3, rate=self.dropout)

        dc4 = self.output_activation(deconv2d(dc3, [self.batch_size, s_h, s_w, 3], 3, 3, 1, 1, name='dc4', stddev=0.02))
        if self.dropout:
          dc4 = tf.layers.dropout(dc4, rate=self.dropout)

      else:
        l0  = self.hidden_activation(linear(z, 4 * 4 * 512, name='l0', stddev=0.02))
        l0  = tf.reshape(l0, [self.batch_size, 4, 4, 512])
        dc1 = self.hidden_activation(deconv2d(l0, [self.batch_size, 8, 8, 256], name='dc1', stddev=0.02))
        dc2 = self.hidden_activation(deconv2d(dc1, [self.batch_size, 16, 16, 128], name='dc2', stddev=0.02))
        dc3 = self.hidden_activation(deconv2d(dc2, [self.batch_size, 32, 32, 64], name='dc3', stddev=0.02))
        dc4 = self.output_activation(deconv2d(dc3, [self.batch_size, 32, 32, 3], 3, 3, 1, 1, name='dc4', stddev=0.02))

      x = dc4
    return x

  def generate_noise(self):
    if self.z_distribution == 'normal':
      return np.random.randn(self.batch_size, self.hidden_dim).astype(np.float32)
    elif self.z_distribution == 'uniform' :
      return np.random.uniform(-1, 1, (self.batch_size, self.hidden_dim)).astype(np.float32)
    else:
      raise NotImplementedError


class SNDCGAN_Discrminator(object):
  def __init__(self, batch_size=64, hidden_activation=lrelu, output_dim=1, scope='critic', spectral_normed=True, noise=None, dropout=None, **kwargs):
    self.batch_size = batch_size
    self.hidden_activation = hidden_activation
    self.output_dim = output_dim
    self.scope = scope
    self.spectral_normed = spectral_normed
    self.noise = noise
    self.dropout = dropout

  def __call__(self, x, y=None, z=None, update_collection=tf.GraphKeys.UPDATE_OPS, **kwargs):
    with tf.variable_scope(self.scope):

      if self.noise:
        x = gaussian_noise_layer(x, stddev=self.noise)

      if self.dropout:
        x = tf.layers.dropout(x, rate=self.dropout)

      c0_0 = conv2d(   x,  64, 3, 3, 1, 1, spectral_normed=self.spectral_normed, update_collection=update_collection, stddev=0.02, name='c0_0')
      if y is not None:
        c0_0 = tf.concat([c0_0, tf.tile(tf.reshape(y, [-1, 1, 1, y.get_shape().as_list()[-1]]),
                                                                [1, tf.shape(c0_0)[1], tf.shape(c0_0)[2], 1])], axis=3)
      c0_0 = self.hidden_activation(c0_0)

      if self.dropout:
        c0_0 = tf.layers.dropout(c0_0, rate=self.dropout)

      c0_1 = self.hidden_activation(conv2d(c0_0, 128, 4, 4, 2, 2, spectral_normed=self.spectral_normed, update_collection=update_collection, stddev=0.02, name='c0_1'))
      if self.dropout:
        c0_1 = tf.layers.dropout(c0_1, rate=self.dropout)

      c1_0 = self.hidden_activation(conv2d(c0_1, 128, 3, 3, 1, 1, spectral_normed=self.spectral_normed, update_collection=update_collection, stddev=0.02, name='c1_0'))
      if self.dropout:
        c1_0 = tf.layers.dropout(c1_0, rate=self.dropout)
      c1_1 = self.hidden_activation(conv2d(c1_0, 256, 4, 4, 2, 2, spectral_normed=self.spectral_normed, update_collection=update_collection, stddev=0.02, name='c1_1'))
      if self.dropout:
        c1_1 = tf.layers.dropout(c1_1, rate=self.dropout)

      c2_0 = self.hidden_activation(conv2d(c1_1, 256, 3, 3, 1, 1, spectral_normed=self.spectral_normed, update_collection=update_collection, stddev=0.02, name='c2_0'))
      if self.dropout:
        c2_0 = tf.layers.dropout(c2_0, rate=self.dropout)
      c2_1 = self.hidden_activation(conv2d(c2_0, 512, 4, 4, 2, 2, spectral_normed=self.spectral_normed, update_collection=update_collection, stddev=0.02, name='c2_1'))
      if self.dropout:
        c2_1 = tf.layers.dropout(c2_1, rate=self.dropout)

      c3_0 = self.hidden_activation(conv2d(c2_1, 512, 3, 3, 1, 1, spectral_normed=self.spectral_normed, update_collection=update_collection, stddev=0.02, name='c3_0'))
      c3_0 = tf.reshape(c3_0, [self.batch_size, -1])
      if y is not None:
        c3_0 = tf.concat(axis=1, values=[c3_0, y])
      if self.dropout:
        c3_0 = tf.layers.dropout(c3_0, rate=self.dropout)

      l4 = linear(c3_0, self.output_dim, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='l4')
      if self.dropout:
        l4 = tf.layers.dropout(l4, rate=self.dropout)

    return tf.reshape(l4, [-1])
