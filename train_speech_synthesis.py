# -*- coding: utf-8 -*-
#/usr/bin/python2

'''
    File name: train_speech_synthesis.py
    Author: Gert-Jan Wille <hello@gert-janwille.com>
    Date created: 09/24/2018
    Date last modified: 10/06/2018
'''

import os
import tensorflow as tf
from scipy.io.wavfile import write

from lib.utils import *
import lib.hyperparams as hp
from lib.data_manager import DataManager, split_data
from lib.logger import initialize, message
# from networks.LSTM import LSTM

initialize()


data_loader = DataManager(dataset_name='synthesis')
fpaths, text_lengths, texts = data_loader.get_data()

maxlen, minlen = max(text_lengths), min(text_lengths)

X_train, X_test, y_train, y_test = split_data(fpaths, texts)

print("X Validation", len(X_train[0]))
print("y Validation", y_train[0])

# mel, mag = get_spectrograms(fpaths[0])
# print(mel, mag)
#
# audio = spectrogram2wav(mag)
# write(os.path.join('./examples/samples', '{}.wav'.format(1)), hp.sr, audio)


# model = LSTM({
#     'input': 1,
#     'decode': 3
# })
