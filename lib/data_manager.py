'''
    Author: Gert-Jan Wille <hello@gert-janwille.com>
    Date created: 10/06/2018
    Date last modified: 10/06/2018
'''

import os
import re
import codecs
import numpy as np
import unicodedata
import lib.hyperparams as hp
from lib.utils import get_spectrograms

def split_data(X, y, validation_split=.2, setName=("Training Set", "Test Set")):
    num_train_samples = int((1 - validation_split)*len(X))
    X_train, y_train, X_test, y_test = X[:num_train_samples], y[:num_train_samples], X[num_train_samples:], y[num_train_samples:]
    print("\t %s: %s %s" % (setName[0], len(X_train), len(y_train)))
    print("\t %s: %s %s" % (setName[1], len(X_test), len(y_test)))
    return X_train, X_test, y_train, y_test

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate("PE abcdefghijklmnopqrstuvwxyz'.?")}
    idx2char = {idx: char for idx, char in enumerate("PE abcdefghijklmnopqrstuvwxyz'.?")}
    return char2idx, idx2char

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents
    text = text.lower()
    text = re.sub("[^{}]".format("PE abcdefghijklmnopqrstuvwxyz'.?"), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text


class DataManager(object):

    def __init__(self, dataset_name=None, dataset_path=None):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'synthesis':
            self.dataset_path = hp.PRIVATE_DATA_PATH + '/voice/Test-1.0'
        else:
            raise Exception('Incorrect dataset name')


    def get_data(self):
        print("Initializing " + self.dataset_name + " Dataset")
        data = None

        if self.dataset_name == 'synthesis':
            data = self._load_synthesis()

        return data if data else []


    def _load_synthesis(self):
        # Load vocabulary
        char2idx, idx2char = load_vocab()
        wavs, text_lengths, texts = [], [], []

        transcript = os.path.join(self.dataset_path, 'metadata.csv')
        lines = codecs.open(transcript, 'r', 'utf-8').readlines()

        for line in lines:
            fname, text = line.strip().split("|")

            fpath = os.path.join(self.dataset_path , "wavs", fname + ".wav")
            wavs.append(get_spectrograms(fpath))

            text = text_normalize(text) + "E" # E: EOS
            text = [char2idx[char] for char in text]

            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32).tostring())
            # texts.append(text)

        return wavs, text_lengths, texts
