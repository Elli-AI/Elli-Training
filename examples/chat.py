'''
    File name: chat.py
    Author: Gert-Jan Wille <hello@gert-janwille.com>
    Date created: 09/24/2018
    Date last modified: 09/24/2018
    Python Version: 3.6
'''

import json
import numpy as np
import random
from keras.models import load_model

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open('../data/intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)

        words.extend(w)

        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))


model = load_model('../models/Elli-Model.h5')

def bow(sentence, words, show_details=False):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

context = {}
ERROR_THRESHOLD = 0.65

def classify(sentence):
    p = bow(sentence, words)

    d = len(p)
    f = len(documents)-2
    a = np.zeros([f, d])
    tot = np.vstack((p,a))

    results = model.predict(tot)[0]
    results = [[i,r] for i,r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))

    return return_list

def response(sentence, userID, show_details=False):
    results = classify(sentence)
    if show_details: print('Result:',results)

    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:

                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])

                        return (random.choice(i['responses']))
            results.pop(0)



while True:
    m = input("> ")
    print(response(m, '#21'))
