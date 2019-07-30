import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

firstfile = "theprince.txt"
raw_text = open(firstfile).read()
raw_text = raw_text.lower()

vect = sorted(list(set(raw_text)))
vect_int = dict((c, i) for i, c in enumerate(vect))

num_vect = len(raw_text)
num_voc = len(vect)

print "total char: ", num_vect
print "total voc: ", num_voc

sequence_len = 100
x_data = []
y_data = []
for index in range(0, num_vect - sequence_len):
	in_sequence = raw_text[index:index + sequence_len]
	out_sequence = raw_text[index + sequence_len]
	x_data.append([vect_int[char] for char in in_sequence])
	y_data.append(vect_int[out_sequence])
num_patt = len(x_data)
print "Total patterns: ", num_patt

X = numpy.reshape(x_data, (num_patt, sequence_len, 1))
X = X / float(num_voc)
y = np_utils.to_categorical(y_data)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

path = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
list_cback = [checkpoint]

#model.fit(X, y, epochs=20, batch_size=128, callbacks=list_cback)
filename = "weights-improvement-20-2.1080.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = numpy.random.randint(0, len(x_data)-1)
pattern = x_data[start]
print "Seed:"
print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print "\nDone."