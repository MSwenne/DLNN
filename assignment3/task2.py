from __future__ import print_function
from generateData import *
from colorama import init
from constants import *
from model import Model
import numpy as np
init()

# Parameters for the model and dataset.
DIGITS = 10

# Maximum length of input is 'bin + bin' (e.g., '101+1010'). Maximum length of
# bin is DIGITS.
MAXLEN = DIGITS + 1 + DIGITS

# All the numbers, plus sign and space for padding.
chars =  '01+ '
ctable = CharacterTable(chars)

generator = DataGenerator(DIGITS, MAXLEN, chars, ctable)

def task2(reverse):
    x_train, y_train, x_val, y_val = generator.generate_training_data(reverse=reverse, binary=True)
    model = Model(DIGITS, MAXLEN, chars)
    model.train(x_train, y_train, x_val, y_val, ctable, reverse)

    all_x, all_y = generator.generate_million_data(reverse=reverse, binary=True)
    # Test the model on all 1 million examples
    for ind, _ in enumerate(all_y):
        rowx, rowy = all_x[np.array([ind])], all_y[np.array([ind])]
        preds = model.model.predict_classes(rowx, verbose=0)
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        if correct == guess:
            count += 1
        MAE += abs(correct-guess)
        MSE += pow(correct-guess,2)
    count = count/len(all_y)
    MAE = MAE/len(all_y)
    MSE = MSE/len(all_y)
    print(f'Task 2 - reverse={reverse}\n\tPercentage: {count}, MAE: {MAE}, MSE: {MSE}')

# Run task 2 normal
task2(False)

# Run task 2 reverse
task2(True)