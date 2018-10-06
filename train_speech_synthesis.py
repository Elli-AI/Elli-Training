# -*- coding: utf-8 -*-
#/usr/bin/python2

'''
    File name: train_speech_synthesis.py
    Author: Gert-Jan Wille <hello@gert-janwille.com>
    Date created: 09/24/2018
    Date last modified: 10/06/2018
'''

import os

import lib.hyperparams as hp
from lib.data_manager import DataManager
from lib.logger import initialize, message

initialize()


data_loader = DataManager(dataset_name='synthesis')
print(data_loader.get_data())
