import os
import sys
import datetime

import logging
import os.path

W = '\033[0m'    # white (normal)
R = '\033[31m'   # red
G = '\033[32m'   # green
O = '\033[33m'   # orange
B = '\033[34m'   # blue
P = '\033[35m'   # purple
C = '\033[36m'   # cyan
GR = '\033[37m'  # gray
T = '\033[93m'   # tan
B = '\033[1m' # bold

import sys

class Logger(object):
    def __init__(self, dir):
        self.terminal = sys.stdout
        self.log = open(dir + "/Elli-AI-TRAINING.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def initialize_logger(dir):
    sys.stdout = Logger(dir)

def initialize():
    print(chr(27) + "[2J")

    x = datetime.datetime.now()
    title = ("%d:%d:%d-%d.%d.%d" % (x.year, x.month, x.day, x.hour, x.minute, x.second))
    directory = './logs/' + title

    if not os.path.exists(directory):
        os.makedirs(directory)

    initialize_logger(directory)
    header(title)

def message(m):
    print("[" + T + "*" + W + "] " + m + "\n")

def userInput(m):
    print("[" + B + ">" + W + "] " + m + "\n")

def warning(m):
    print("[" + R + "!" + W + "] WARNING: " + m + "\n")

def header(headline):
    print(B + T + '\nELLI - AI (training)' + W + " - " + headline + "\n\n")
