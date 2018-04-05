import numpy as np
import os
from glob import glob
import scipy.misc
import pandas as pd
import ast

class DataSet(object):
  def __init__(self, batch_size=64, data_dir='', labels_file=None, labels_names=None, file_mask='*.jpg', resize=None):
    files = glob(os.path.join(data_dir, file_mask))

    if resize:
      images = np.array([scipy.misc.imresize(scipy.misc.imread(batch_file), resize).astype(np.float) for batch_file in files]).astype(np.float32)
    else:
      images = np.array([scipy.misc.imread(batch_file).astype(np.float) for batch_file in files]).astype(np.float32)
    self.images = (images - 127.5) / 127.5

    labels = None
    if labels_file is not None:
      labels_df = pd.read_csv(data_dir + "/" + labels_file, index_col=0, delim_whitespace=True)
      labels_df[labels_df == -1] = 0
      if labels_names:
        labels_df = labels_df[ast.literal_eval(labels_names)]
      labels = np.array([labels_df.loc[os.path.basename(X)].tolist() for X in files], np.float)
    self.labels = labels

    self.batch_size = batch_size
    self.num_samples = len(self.images)
    self.reset()

  def reset(self):
    self.shuffle_samples()
    self.next_batch_pointer = 0

  def shuffle_samples(self):
    shuffled_indices = np.random.permutation(np.arange(self.num_samples))
    self.images = self.images[shuffled_indices]
    if self.labels is not None:
      self.labels = self.labels[shuffled_indices]

  def has_next_batch(self):
    num_samples_left = self.num_samples - self.next_batch_pointer
    return num_samples_left >= self.batch_size

  def get_next_batch(self):
    if not self.has_next_batch():
      self.reset()
    batch = self.images[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
    if self.labels is not None:
      labels = self.labels[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
    self.next_batch_pointer += self.batch_size
    if self.labels is not None:
      return batch, labels
    else:
      return batch

  def get_all(self):
    if self.labels is not None:
      return self.images, self.labels
    else:
      return self.images
