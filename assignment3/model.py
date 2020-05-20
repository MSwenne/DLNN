from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np
from constants import *

class Model():
    def __init__(self, digits, maxlen, chars, pairs):
        print('Build model...')
        self.model = Sequential()
        # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
        # Note: In a situation where your input sequences have a variable length,
        # use input_shape=(None, num_feature).
        if pairs:
            self.model.add(RNN(HIDDEN_SIZE, input_shape=(digits, len(chars))))
        else:
            self.model.add(RNN(HIDDEN_SIZE, input_shape=(maxlen, len(chars))))
        # As the decoder RNN's input, repeatedly provide with the last output of
        # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
        # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
        self.model.add(layers.RepeatVector(digits + 1))
        # The decoder RNN could be multiple layers stacked or a single layer.
        for _ in range(LAYERS):
            # By setting return_sequences to True, return not only the last output but
            # all the outputs so far in the form of (num_samples, timesteps,
            # output_dim). This is necessary as TimeDistributed in the below expects
            # the first dimension to be the timesteps.
            self.model.add(RNN(HIDDEN_SIZE, return_sequences=True))

        # Apply a dense layer to the every temporal slice of an input. For each of step
        # of the output sequence, decide which character should be chosen.
        self.model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
        self.model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])
        self.model.summary()

    def train(self, x_train, y_train, x_val, y_val, ctable, reverse):
        for iteration in range(1, 200):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            self.model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=1,
                    validation_data=(x_val, y_val))
            # Select 10 samples from the validation set at random so we can visualize
            # errors.
            for _ in range(10):
                ind = np.random.randint(0, len(x_val))
                rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
                preds = self.model.predict_classes(rowx, verbose=0)
                q = ctable.decode(rowx[0])
                correct = ctable.decode(rowy[0])
                guess = ctable.decode(preds[0], calc_argmax=False)
                print('Q', q[::-1] if reverse else q, end=' ')
                print('T', correct, end=' ')
                if correct == guess:
                    print(f'{colors.ok}☑{colors.close}', end=' ')
                else:
                    print(f'{colors.fail}☒{colors.close}', end=' ')
                print(guess)