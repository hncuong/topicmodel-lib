import logging
from .base import Corpus, DataIterator


class DataSet(DataIterator):
    def __init__(self, data_path, batch_size=None, passes=1, shuffle_every=None):
        """
        A class for loading mini-batches of data from a file
        Args:
            data_path: path to file, type(str)
            batch_size: size of a mini-batch
            passes: number of passes over the origin data
            shuffle_every: shuffle the origin data after a number of passes
        """
        super(DataSet, self).__init__()
        self.data_path = data_path
        self.passes = passes
        self.no_pass = 0
        self.no_iter = 0
        self.temp_path = self.data_path

    def load_mini_batch(self):
        pass

    def end_of_data(self):
        pass