import numpy as np
from characterTable import CharacterTable
from model import Model
from constants import *
import random

class DataGenerator():
    def __init__(self, digits, maxlen, chars, ctable):
        self.digits = digits
        self.maxlen = maxlen
        self.chars = chars
        self.ctable = ctable
        self.seen = set()

    def generate_number(self, reverse):
        # Normal is maxlen == 3, binary if maxlen == 10
        maxnumber = 1000 if self.maxlen == 3 else 1024
        while True:
            a = random.randint(0,maxnumber)
            b = random.randint(0,maxnumber)
            q = '{}+{}'.format(a, b)
            # Pad the data with spaces such that it is always MAXLEN.
            query = q + ' ' * (self.maxlen - len(q))
            if reverse:
                # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
                # space used for padding.)
                query = query[::-1]
            # Skip any addition questions we've already seen
            # Also skip any such that x+Y == Y+x (hence the sorting).
            # Answers can be of maximum size DIGITS + 1.
            key = tuple(sorted((a, b)))
            if not (key in self.seen):
                self.seen.add(key)
                return a, b, query

    def generate_training_data(self, reverse, binary):
        questions = []
        expected = []
        print('Generating data...')
        while len(questions) < TRAINING_SIZE:
            a, b, query = self.generate_number(reverse)
            if binary:
                ans = int(a) + int(b)
                ans = bin(ans)[2:]
                print(int(a),int(b), ans)
            else:
                ans = str(a + b)
            ans += ' ' * (self.digits + 1 - len(ans))
            questions.append(query)
            expected.append(ans)
        print('Total addition questions:', len(questions))

        print('Vectorization...')
        x = np.zeros((len(questions), self.maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(questions), self.digits + 1, len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(questions):
            x[i] = self.ctable.encode(sentence, self.maxlen)
        for i, sentence in enumerate(expected):
            y[i] = self.ctable.encode(sentence, self.digits + 1)

        # Shuffle (x, y) in unison as the later parts of x will almost all be larger
        # digits.
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

        # Explicitly set apart 10% for validation data that we never train over.
        split_at = len(x) - len(x) // 10
        (x_train, x_val) = x[:split_at], x[split_at:]
        (y_train, y_val) = y[:split_at], y[split_at:]

        print('Training Data:')
        print(x_train.shape)
        print(y_train.shape)

        print('Validation Data:')
        print(x_val.shape)
        print(y_val.shape)
        return x_train, y_train, x_val, y_val

    def generate_million_data(self, reverse, binary):
        all_numbers = []
        all_questions = []
        all_expected = []
        print('Generating 1 million datapoints...')
        for i in range(1000):
            all_numbers.append(i)
        for i in all_numbers:
            for j in all_numbers:
                q = '{}+{}'.format(i, j)
                query = q + ' ' * (self.maxlen - len(q))
                ans = str(i + j)
                # Answers can be of maximum size DIGITS + 1.
                ans += ' ' * (self.digits + 1 - len(ans))
                if reverse:
                    # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
                    # space used for padding.)
                    query = query[::-1]
                all_questions.append(query)
                all_expected.append(ans)

        print('Vectorization...')
        all_x = np.zeros((len(all_questions), self.maxlen, len(self.chars)), dtype=np.bool)
        all_y = np.zeros((len(all_questions), self.digits + 1, len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(all_questions):
            all_x[i] = self.ctable.encode(sentence, self.maxlen)
        for i, sentence in enumerate(all_expected):
            all_y[i] = self.ctable.encode(sentence, self.digits + 1)
        return all_x, all_y