import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from datetime import datetime

from charactertable import CharacterTable
from keras.callbacks import ModelCheckpoint
import itertools
import csv



gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class AdderExperimentFactory:
    def __init__(self, reversed, binary, paired):
        self.reversed = reversed
        self.binary = binary
        self.paired = paired
        self.digits = 10 if binary else 3
        self.base = 2 if binary else 10
        self.alphabet = "".join(str(d) for d in range(self.base))
        if not self.paired: self.alphabet += "+"
        self.char_table = CharacterTable(self.alphabet)
        self.input_len = 2*self.digits if self.paired else 2*self.digits + 1
        self.output_len = self.digits + 1

    def generate_model(self, rnn=layers.LSTM, hidden_size=128, num_layers=1):
        model = Sequential()
        model.add(rnn(hidden_size, input_shape=(self.input_len, len(self.alphabet))))
        model.add(layers.RepeatVector(self.output_len))
        for _ in range(num_layers):
            model.add(rnn(hidden_size, return_sequences=True))
        model.add(layers.TimeDistributed(layers.Dense(len(self.alphabet), activation='softmax')))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def generate_full_dataset(self):
        n = self.base**self.digits
        domain = np.arange(n)
        X = []
        y = []

        for i, (a, b) in enumerate(product(domain, repeat=2)):
            fmt = "{:0{digits}{flag}}"
            flag = "b" if self.binary else ""
            a_digits = fmt.format(a, digits=self.digits, flag=flag)
            b_digits = fmt.format(b, digits=self.digits, flag=flag)
            ans_digits = fmt.format(a + b, digits=self.digits+1, flag=flag)

            if self.paired:
                query = "".join(a+b for a, b in zip(a_digits, b_digits))
            else:
                query = a_digits + "+" + b_digits

            if self.reversed:
                query = query[::-1]

            X.append(self.char_table.encode(query, self.input_len))
            y.append(self.char_table.encode(ans_digits, self.output_len))

        return np.array(X), np.array(y)

    def compute_errors(self, y_true, y_pred):
        y_true_decoded = np.array([self.toint(self.char_table.decode(y)) for y in y_true])
        y_pred_decoded = np.array([self.toint(self.char_table.decode(y)) for y in y_pred])
        acc = np.mean(y_true_decoded == y_pred_decoded)
        mse = np.mean(np.square(y_true_decoded - y_pred_decoded))
        mae = np.mean(np.abs(y_true_decoded - y_pred_decoded))
        return {"acc": acc, "mse": mse, "mae": mae}

    def toint(self, x):
        try:
            return int(x, self.base)
        except ValueError:
            return 0


def run_experiment(rnn, reversed, binary, paired, seed=42):
    reversed_folder = "reversed" if reversed else "normal"
    binary_folder = "binary" if binary else "integer"
    paired_folder = "paired" if paired else "not_paired"
    filepath="./checkpoints/" + rnn.__qualname__ + "/" + reversed_folder + "_" + binary_folder + "_" + paired_folder + ""
    logdir = "logs/scalars/" + rnn.__qualname__ + "_" + reversed_folder + "_" + binary_folder + "_" + paired_folder + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        mode='auto', save_freq='epoch',
        save_best_only=False
    )

    callbacks_list = [model_checkpoint_callback, tensorboard_callback]

    fac = AdderExperimentFactory(reversed, binary, paired)
    X, y = fac.generate_full_dataset()
    # Use 5% for training.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=seed)
    # Also only use 5% for validation.
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.95, random_state=seed ^ 0xdeadbeef)

    model = fac.generate_model(rnn)
    history = model.fit(X_train, y_train, batch_size=128, epochs=200, validation_data=(X_val, y_val), callbacks=callbacks_list)
    train_error = fac.compute_errors(y_train, model.predict(X_train))
    full_dataset_error = fac.compute_errors(y, model.predict(X))
    with open(f'./results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([rnn.__qualname__, reversed, binary, paired, train_error, full_dataset_error])
    return train_error, full_dataset_error, history


if __name__ == "__main__":
    networks = [
        layers.LSTM, 
        layers.GRU,
        layers.SimpleRNN
        ]
    # Reverse, binary, paired
    l=[False,True]
    options = list(itertools.product(l,repeat=3))

    for network in networks:
        for (reverse, binary, paired) in options:
            results = run_experiment(rnn=network, reversed=reverse, binary=binary, paired=paired)
