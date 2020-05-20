from tensorflow.keras import layers

# Parameters for the model and dataset.
TRAINING_SIZE = 10

# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'