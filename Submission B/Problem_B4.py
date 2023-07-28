# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.91 and logs.get('val_accuracy') > 0.91):
            self.model.stop_training = True
def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')
    bbc.head()

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    sentences = bbc['text']
    labels = bbc['category']

    sentences_train, sentences_test, labels_train, labels_test = train_test_split(sentences, labels,
                                                                                  train_size=training_portion,
                                                                                 shuffle=False)

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences_train)
    word_index = tokenizer.word_index

    train_seq = tokenizer.texts_to_sequences(sentences_train)
    test_seq = tokenizer.texts_to_sequences(sentences_test)

    train_seq_pad = pad_sequences(train_seq,
                                  maxlen=max_length,
                                  truncating=trunc_type,
                                  padding=padding_type)

    test_seq_pad = pad_sequences(test_seq,
                                 maxlen=max_length,
                                 truncating=trunc_type,
                                 padding=padding_type)
    # You can also use Tokenizer to encode your label.
    labels_tokenizer = Tokenizer()
    labels_tokenizer.fit_on_texts(labels)

    labels_train = np.array(labels_tokenizer.texts_to_sequences(labels_train))
    labels_test = np.array(labels_tokenizer.texts_to_sequences(labels_test))

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size,
                                  embedding_dim,
                                  input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.summary()

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    callback = myCallback()

    model.fit(train_seq_pad,
              labels_train,
              epochs=100,
              validation_data=(test_seq_pad, labels_test),
              callbacks=[callback])

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
