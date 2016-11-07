import logging
import base
from .base import Corpus, DataIterator, DataFormat


class DataSet(DataIterator):
    def __init__(self, data_path, batch_size=None, passes=1, shuffle_every=None,
                 vocab_file=None):
        """
        A class for loading mini-batches of data from a file
        Args:
            data_path: path to file, type(str)
            batch_size: size of a mini-batch
            passes: number of passes over the origin data
            shuffle_every: shuffle the origin data after a number of passes
        """
        super(DataSet, self).__init__()
        # pre process data if needed
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        input_format = base.check_input_format(data_path)
        if input_format == DataFormat.RAW_TEXT:
            vocab_file, tf_file, sq_file = base.pre_process(data_path)
            self.data_path = tf_file
            self.vocab_file = vocab_file
            self.data_format = DataFormat.TERM_FREQUENCY
        else:
            self.data_path = data_path
            self.data_format = input_format
        self.output_format = DataFormat.TERM_FREQUENCY

        self.passes = passes
        self.shuffle_every = shuffle_every

        self.pass_no = 1
        self.batch_no_in_pass = 0

        self.work_path = self.data_path
        self.fp = None
        self.end_of_file = False

    def load_mini_batch(self):
        """

        Returns:

        """
        if self.end_of_file:
            # reset batch no and increase pass no
            self.batch_no_in_pass = 0
            self.pass_no += 1
            logging.info('Pass no: %s', self.pass_no)
            # shuffle after number of passes
            if self.shuffle_every > 0 and self.pass_no % self.shuffle_every == 0:
                self.work_path = base.shuffle_formatted_data_file(self.data_path)

        if self.batch_no_in_pass == 0:
            self.fp = open(self.work_path, 'r')

        logging.info("Mini batch no: %s", self.batch_no_in_pass)

        mini_batch, end_of_file = self.load_mini_batch_and_state_with_format()
        if end_of_file:
            self.end_of_file = end_of_file
            self.fp.close()
        self.batch_no_in_pass += 1
        self.mini_batch_no += 1

        return mini_batch

    def load_mini_batch_and_state_with_format(self):
        """

        Returns:

        """
        if self.output_format == DataFormat.TERM_FREQUENCY:
            if self.data_format == DataFormat.TERM_FREQUENCY:
                return base.load_mini_batch_term_frequency_from_term_frequency_file(self.fp, self.batch_size)
            else:
                return base.load_mini_batch_term_frequency_from_sequence_file(self.fp, self.batch_size)
        else:
            if self.data_format == DataFormat.TERM_FREQUENCY:
                return base.load_mini_batch_term_sequence_from_term_frequency_file(self.fp, self.batch_size)
            else:
                return base.load_mini_batch_term_sequence_from_sequence_file(self.fp, self.batch_size)

    def end_of_data(self):
        if self.end_of_file and self.pass_no == self.passes:
            self.end_of_data = True
        return self.end_of_data

    def set_output_format(self, output_format):
        assert (output_format == DataFormat.TERM_SEQUENCE or output_format == DataFormat.TERM_FREQUENCY), \
            'Corpus format type must be term-frequency (tf) or sequences (sq)!'
        self.output_format = output_format
