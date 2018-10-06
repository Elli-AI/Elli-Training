from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Embedding, Bidirectional
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM

class LSTM(object):
    def __init__(self, args):
        self.args = args
        print(self.args)

        self.model = Sequential()

    # def encoder(self):
    #     model.add(Embedding(input_vocab_size+2, hidden, input_length=X_seq_len, mask_zero=True))
    #     print('Embedding layer created')
    #     model.add(Bidirectional(LSTM(hidden, return_sequences = True), merge_mode = 'concat'))
    #     model.add(Bidirectional(LSTM(hidden, return_sequences = True), merge_mode = 'concat'))
    #     model.add(Bidirectional(LSTM(hidden), merge_mode = 'concat'))
    #     model.add(RepeatVector(y_seq_len))
    #     print('Encoder layer created')
    #
    # def decoder(self):
    #     for _ in range(layers):
    #         model.add(LSTM(hidden, return_sequences=True))
    #     model.add(TimeDistributed(Dense(target_vocab_size+1)))
    #     model.add(Activation('softmax'))
    #     model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #     print('Decoder layer created')
