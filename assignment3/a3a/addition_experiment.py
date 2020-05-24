import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

from charactertable import CharacterTable



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
                query = "".join(zip(a_digits, b_digits))
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
    fac = AdderExperimentFactory(reversed, binary, paired)
    X, y = fac.generate_full_dataset()
    # Use 5% for training.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=seed)
    # Also only use 5% for validation.
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.95, random_state=seed ^ 0xdeadbeef)

    model = fac.generate_model(rnn)
    history = model.fit(X_train, y_train, batch_size=128, epochs=1, validation_data=(X_val, y_val))
    train_error = fac.compute_errors(y_train, model.predict(X_train))
    full_dataset_error = fac.compute_errors(y, model.predict(X))
    return train_error, full_dataset_error, history


if __name__ == "__main__":
    print(run_experiment(rnn=layers.LSTM, reversed=False, binary=True, paired=False))
