import timeit

import numpy as np
import tensorflow as tf

import os
from libs.input_helper import DataSet
from libs.utils import save_images, mkdir
from net import DCGANGenerator, SNDCGAN_Discrminator

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/data/img_align_celeba', 'data dir')
flags.DEFINE_integer('image_size_height', 64, 'pixels')
flags.DEFINE_integer('image_size_width', 52, 'pixels')

flags.DEFINE_string('labels_file', "list_attr_celeba.txt", 'set to None to ignore (turn off conditioning)')
flags.DEFINE_string('labels_names', "['Male','Smiling','Black_Hair','Blond_Hair','Mustache']", 'set to None to use all')

flags.DEFINE_string('data_name', 'celeba_conditioned', 'Experiment name, folder will be created in this name')
flags.DEFINE_string('output_dir', 'output', 'Directory that contains all outputs')
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_integer('display_interval', 1, 'interval of displaying log to console')
flags.DEFINE_integer('iterations', 100000, '')

flags.DEFINE_float('d_dropout', 0.5, 'Dropout rate for D. Set to None to turn off dropout')
flags.DEFINE_float('d_gaussian_noise_stddev', 0.5, 'stddev for gaussian noise added to D inputs. Set to None to turn off noise')
flags.DEFINE_float('d_one_sided_label_smooth', 0.9, 'None to not adjust')

flags.DEFINE_integer('snapshot_interval', 2500, 'When to save checkpoints and samples as training progresses')
flags.DEFINE_float('adam_alpha', 0.0002, 'learning rate')
flags.DEFINE_float('adam_beta1', 0.5, 'beta1 in Adam')
flags.DEFINE_float('adam_beta2', 0.999, 'beta2 in Adam')

output_dir = FLAGS.output_dir+"/"+FLAGS.data_name
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

config = FLAGS.__flags
print('FLAGS:',config)

generator = DCGANGenerator(**config)
discriminator = SNDCGAN_Discrminator(**config)

print("Loading images...")
data_set = DataSet(batch_size=FLAGS.batch_size,
               data_dir=FLAGS.data_dir,
               file_mask="*.jpg",
               labels_file=FLAGS.labels_file,
               labels_names=FLAGS.labels_names,
               resize=(FLAGS.image_size_height, FLAGS.image_size_width)
               )
print("Images loaded")

y_dim = None
if FLAGS.labels_file:
    y_dim = (data_set.get_all())[1].shape[1]

global_step = tf.Variable(0, name="global_step", trainable=False)
increase_global_step = global_step.assign(global_step + 1)
is_training = tf.placeholder(tf.bool, shape=())

y = None
if y_dim:
    y = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, generator.generate_noise().shape[1]])
x_hat = generator(z, y, is_training=is_training)
x = tf.placeholder(tf.float32, shape=x_hat.shape)

d_fake = discriminator(x_hat, y, z, noise=FLAGS.d_gaussian_noise_stddev, dropout=FLAGS.d_dropout, update_collection="None")
d_real = discriminator(x, y, z, noise=FLAGS.d_gaussian_noise_stddev, dropout=FLAGS.d_dropout, update_collection="NO_OPS")
if FLAGS.d_one_sided_label_smooth:
    d_labels_real = tf.ones_like(d_real) * tf.random_uniform(shape=tf.shape(d_real), minval=FLAGS.d_one_sided_label_smooth, maxval=1.0, dtype=tf.float32)
    d_labels_fake = tf.zeros_like(d_fake)
    d_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=d_labels_fake)
        +
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=d_labels_real))
else:
    d_loss = tf.reduce_mean(tf.nn.softplus(d_fake) + tf.nn.softplus(-d_real))

if FLAGS.d_one_sided_label_smooth:
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake,
            labels=tf.ones_like(d_fake)
        )
    )
else:
    g_loss = tf.reduce_mean(tf.nn.softplus(-d_fake))
d_loss_summary_op = tf.summary.scalar('d_loss', d_loss)
g_loss_summary_op = tf.summary.scalar('g_loss', g_loss)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(output_dir+'/snapshots')

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.adam_alpha, beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2)
d_gvs = optimizer.compute_gradients(d_loss, var_list=d_vars)
g_gvs = optimizer.compute_gradients(g_loss, var_list=g_vars)
d_solver = optimizer.apply_gradients(d_gvs)
g_solver = optimizer.apply_gradients(g_gvs)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=0)
if tf.train.latest_checkpoint(output_dir+'/snapshots') is not None:
  saver.restore(sess, tf.train.latest_checkpoint(output_dir+'/snapshots'))

np.random.seed(1337)
sample_z = generator.generate_noise()
if y_dim:
    sample_y = np.random.randint(2, size=(FLAGS.batch_size, y_dim))
    sample_y[0:FLAGS.batch_size // 2, 0] = 1 #Set first half of generated samples to male
    sample_y[FLAGS.batch_size // 2:, 0] = 0  #Set second half of generated samples to female

np.random.seed()
iteration = sess.run(global_step)
start = timeit.default_timer()

is_start_iteration = True
print('Starting at iteration '+str(iteration)+'...')
while iteration < FLAGS.iterations:
  if y_dim:
      images, labels = data_set.get_next_batch()
  else:
      images = data_set.get_next_batch()

  if y_dim:
      _, g_loss_curr = sess.run([g_solver, g_loss], feed_dict={z: generator.generate_noise(), y: labels, is_training: True})
  else:
      _, g_loss_curr = sess.run([g_solver, g_loss],
                                feed_dict={z: generator.generate_noise(), is_training: True})
  if y_dim:
      _, d_loss_curr, summaries = sess.run([d_solver, d_loss, merged_summary_op],
                                             feed_dict={x: images,
                                                        y: labels,
                                                        z: generator.generate_noise(),
                                             is_training: True})
  else:
      _, d_loss_curr, summaries = sess.run([d_solver, d_loss, merged_summary_op],
                                             feed_dict={x: images,
                                                        z: generator.generate_noise(),
                                             is_training: True})

  sess.run(increase_global_step)

  if (iteration + 1) % FLAGS.display_interval == 0 and not is_start_iteration:
    summary_writer.add_summary(summaries, global_step=iteration)
    stop = timeit.default_timer()
    print('Iter {}: d_loss = {:4f}, g_loss = {:4f}, time = {:2f}s'.format(iteration, d_loss_curr, g_loss_curr, stop - start))
    start = stop

  if (iteration + 1) % FLAGS.snapshot_interval == 0 and not is_start_iteration:
    saver.save(sess, output_dir+'/snapshots/model.ckpt', global_step=iteration)
    if y_dim:
        sample_images = sess.run(x_hat, feed_dict={z: sample_z, y: sample_y, is_training: False})
    else:
        sample_images = sess.run(x_hat, feed_dict={z: sample_z, is_training: False})
    save_images(sample_images, output_dir+'/{:06d}.png'.format(iteration))

  iteration += 1
  is_start_iteration = False