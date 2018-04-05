import pandas as pd
import numpy as np
import tensorflow as tf
from net import DCGANGenerator
import scipy.misc
from libs.utils import save_images

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_name', 'celeba_conditioned', '')
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_integer('image_size_height', 64, 'pixels, assuming same length x width')
flags.DEFINE_integer('image_size_width', 52, 'pixels, assuming same length x width')
flags.DEFINE_string('output_dir', 'output', 'Directory that contains all outputs')
save_individual_images = False
save_collage = True
COLS = ['Male','Smiling','Black_Hair','Blond_Hair','Mustache']
y_dim = len(COLS)

if y_dim:
    sample_y = pd.DataFrame(np.zeros(shape=[FLAGS.batch_size, y_dim]), columns=COLS)
    sample_y['Male'] = 1
    sample_y['Smiling'] = 1
    sample_y['Black_Hair'] = 1
    sample_y['Blond_Hair'] = 0
    sample_y['Mustache'] = 1

output_dir = FLAGS.output_dir+"/"+FLAGS.data_name

config = FLAGS.__flags
generator = DCGANGenerator(**config)

global_step = tf.Variable(0, name="global_step", trainable=False)

is_training = tf.placeholder(tf.bool, shape=())
y = None
if y_dim:
    y = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, generator.generate_noise().shape[1]])
x_hat = generator(z, y, is_training=is_training)
x = tf.placeholder(tf.float32, shape=x_hat.shape)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
if tf.train.latest_checkpoint(output_dir+'/snapshots') is not None:
    saver.restore(sess, tf.train.latest_checkpoint(output_dir+'/snapshots'))
else:
    print("Unable to load tensorflow checkpoint for "+FLAGS.data_name)
    exit()

print("---")
print("For '"+FLAGS.data_name+"', loaded tensorflow checkpoint from iteration #",sess.run(global_step))

print("> Generating",FLAGS.batch_size,"images")
sample_z = generator.generate_noise()
if y_dim:
    sample_images = sess.run(x_hat, feed_dict={z: sample_z, y: sample_y, is_training: False})
else:
    sample_images = sess.run(x_hat, feed_dict={z: sample_z, is_training: False})

if save_individual_images:
    for i, image in enumerate(sample_images):
        scipy.misc.imsave(output_dir+"/"+str(i)+".png",image)
if save_collage:
    save_images(sample_images, output_dir+'/collage.png')