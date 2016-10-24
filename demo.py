from tmlib.lda.Online_VB import OnlineVB
from tmlib.datasets.base import Dataset
import sys


def main():
    file_path = 'dataset/tweet_1k.txt'
    data = Dataset(file_path)


if __name__ == '__main__':
    print 'OK'
