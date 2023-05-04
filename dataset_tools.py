import os

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset


def parse_tfrecord_tf(record):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
    data = tf.io.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])

def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)

def parse_tfrecord_torch(record):
    return torch.from_numpy(parse_tfrecord_np(record))


class FaceDataset(Dataset):
    def __init__(self, root, file_name):
        self.root = root
        self.file_name = file_name

        self.raw_dataset = tf.data.TFRecordDataset(os.path.join(root, file_name))
        self.data_iter = self.raw_dataset.enumerate().as_numpy_iterator()

        self.preload_images()

    def preload_images(self):
        self.images = []
        
        for _, record in self.data_iter:
            self.images.append(parse_tfrecord_torch(record))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx] / 127.5 - 1


def get_latent_vector(n_samples, latent_dim):
    """
    Generates a latent vector of shape (n_samples, latent_dim, 1, 1) with
    random values on a `latent_dim`-dimensional hypersphere.
    """

    x_input = torch.randn(n_samples, latent_dim)

    x_input /= torch.linalg.vector_norm(x_input, ord=2, dim=1, keepdims=True)

    return x_input.reshape(n_samples, latent_dim, 1, 1)
