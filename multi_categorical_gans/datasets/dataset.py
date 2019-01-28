from __future__ import division

from future.utils import implements_iterator

import numpy as np


class Dataset(object):

    def __init__(self, features):
        self.features = features

    def split(self, proportion):
        assert 0 < proportion < 1, "Proportion should be between 0 and 1."

        limit = int(np.floor(len(self.features) * proportion))

        return Dataset(self.features[:limit, :]), Dataset(self.features[limit:, :])

    def batch_iterator(self, batch_size, shuffle=True):
        if shuffle:
            indices = np.random.permutation(len(self.features))
        else:
            indices = np.arange(len(self.features))
        return DatasetIterator(self.features, indices, batch_size)


@implements_iterator
class DatasetIterator(object):

    def __init__(self, features, indices, batch_size):
        self.features = features
        self.indices = indices
        self.batch_size = batch_size

        self.batch_index = 0
        self.num_batches = int(np.ceil(len(features) / batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_index >= self.num_batches:
            raise StopIteration
        else:
            batch_start = self.batch_index * self.batch_size
            batch_end = (self.batch_index + 1) * self.batch_size
            self.batch_index += 1
            return self.features[self.indices[batch_start:batch_end]]
