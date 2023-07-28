# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('sarcasm.json', 'r') as f:
        datastore = json.load(f)

    print(datastore[0])
    print(datastore[2])

    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    sentences_train = sentences[:training_size]
    labels_train = labels[:training_size]
    sentences_test = sentences[training_size:]
    labels_test = labels[training_size:]

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size,
                          oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences_train)
    word_index = tokenizer.word_index

    train_seq = tokenizer.texts_to_sequences(sentences_train)
    train_seq_pad = pad_sequences(train_seq,
                                    maxlen=max_length,
                                    truncating=trunc_type,
                                    padding=padding_type
                                    )

    test_seq = tokenizer.texts_to_sequences(sentences_test)
    test_seq_pad = pad_sequences(test_seq,
                                   maxlen=max_length,
                                   truncating=trunc_type,
                                   padding=padding_type
                                   )

    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)

    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )

    model.fit(train_seq_pad,
              labels_train,
              epochs=20,
              validation_data=(test_seq_pad, labels_test),
              verbose=2
              )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4.h5")