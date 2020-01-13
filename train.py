# https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17

import numpy as np
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SpatialDropout1D, LSTM
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


MAX_WORDS = 50000

MAX_SEQUENCE_LENGTH = 200

EMBEDDING_DIM = 100


subs = ["worldnews", "technology", "gaming", "travel"]

# creepy = [line.rstrip('\n') for line in open("clean_creepy.txt")]
gaming = [line.rstrip('\n') for line in open("clean_gaming.txt")]
technology= [line.rstrip('\n') for line in open("clean_technology.txt")]
travel = [line.rstrip('\n') for line in open("clean_travel.txt")]
worldnews = [line.rstrip('\n') for line in open("clean_worldnews.txt")]

allData = worldnews + technology + gaming + travel

Y = np.array([[1, 0, 0, 0] for i in range(len(worldnews))] + [[0, 1, 0, 0] for i in range(len(technology))] \
    + [[0, 0, 1, 0] for i in range(len(gaming))] + [[0, 0, 0, 1] for i in range(len(travel))])

tokenizer = Tokenizer(num_words=MAX_WORDS, filters='`', lower=True)

tokenizer.fit_on_texts(allData)

X = tokenizer.texts_to_sequences(allData)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 404)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(120, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, 
batch_size=batch_size,validation_split=0.1,
callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()

# plt.title('Accuracy')
# plt.plot(history.history['acc'], label='train')
# plt.plot(history.history['val_acc'], label='test')
# plt.legend()
# plt.show()
