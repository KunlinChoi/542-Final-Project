'''
#Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
**Notes**
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import getvector

max_features = 10000
# cut texts after this number of words (among top max_features most common words)
maxlen = 32 #64
batch_size = 64
epo = 200

print('Loading data...')
x_train,x_test,y_train,y_test,moviex,moviey = getvector.getvector()
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(x_train[2])

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

#moviex = sequence.pad_sequences(moviex, maxlen=maxlen)
#print(type(moviex),type(x_train),123)
#print(len(moviex))

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epo,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

#score2, acc2 = model.evaluate(moviex, moviey,
#                            batch_size=batch_size)
#print('Movie Review Test score:', score2)
#print('Movie Review Test accuracy:', acc2)



