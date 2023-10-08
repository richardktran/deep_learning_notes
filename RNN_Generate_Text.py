import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import os

data = open('datasets/text_data/data.txt').read()

corpus = data.lower().split("\n")

# Tokenize the corpus
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

"""
Make sequences of tokens from the corpus like this:
    from fairest
    from fairest creatures
    from fairest creatures we
    from fairest creatures we desire
    from fairest creatures we desire increase
    that thereby
    that thereby beauty's
    that thereby beauty's rose
    that thereby beauty's rose might
    that thereby beauty's rose might never
"""
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to make them all the same length
max_sequence_len = max(len(x) for x in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# 15462 sequences of length 11

# Create predictors and label
predictor, label = input_sequences[:, :-1], input_sequences[:, -1]

# Check results of the above
for i in range(10):
    print("{} => {}".format(" ".join(tokenizer.sequences_to_texts(predictor[i:i + 1])),
                            tokenizer.sequences_to_texts([label[i:i + 1]])
                            )
          )

# Convert label to categorical with one-hot encoding with total_words as the number of classes
label = to_categorical(label, num_classes=total_words)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_len - 1))
# subtract 1 from max_sequence_len because we removed the last word from each sequence to use as the label
# output_dim is the number of dimensions (features) in the embedding vector
model.add(Bidirectional(LSTM(150, return_sequences=True)))
# return_sequences=True because we want to feed the output of this layer into the next layer
# 150 is the number of neurons in the layer
model.add(Dropout(0.2))
model.add(LSTM(100))
# add another LSTM layer with 100 neurons to further refine the features
model.add(Dense(units=total_words / 2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# add a dense layer with half the number of neurons as the input dimension because we want to reduce the number of features to prevent overfitting
model.add(Dense(units=total_words, activation='softmax'))
# add a dense layer with the same number of neurons as the number of classes because we want to output a probability for each class
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

if os.path.exists('models/rnn/text_generator.h5'):
    model = load_model('models/rnn/text_generator.h5')

# Train the model
history = model.fit(predictor, label, epochs=100, verbose=1,
                    callbacks=[
                        ModelCheckpoint('models/rnn/text_generator.h5', save_best_only=True, monitor='accuracy')])

# Load the model

test_sentence = "When that mine eye is famish'd for a"

max_words = 20

for _ in range(max_words):
    token_list = tokenizer.texts_to_sequences([test_sentence])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_id = np.argmax(predicted)
    if predicted_id in tokenizer.index_word:
        output_word = tokenizer.index_word[predicted_id]
        test_sentence += " " + output_word
    else:
        break
print(test_sentence)
