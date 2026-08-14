import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.Base_Neural_Network import Base_Neural_Network
from utils import Optimizers
from utils import Loss_Functions
from utils.Layers import Activation, RNN, LSTM, BatchNormalization
from utils.Data_Functions import Data_Functions

import numpy as np
from matplotlib import pyplot as plt

def generate_geometric_series(size):
    X = np.zeros([size, 10, 18], dtype = float)
    y = np.zeros([size, 10, 18], dtype = float)
    for i in range(size):
        start = np.random.randint(1, 8)
        geom_series = np.geomspace(start = start, stop = start * 512, num = 10)
        X[i] = Data_Functions.to_categorical(geom_series, n_col = 48)
        y[i] = np.roll(X[i], -1, axis = 0)
    y[:, -1, 1] = 1 # Mark end of sequence

def generate_arithmetic_series(size):
    X = np.zeros([size, 10, 18], dtype = float)
    y = np.zeros([size, 10, 18], dtype = float)
    for i in range(size):
        start = np.random.randint(0, 9)
        arith_series = np.arange(start, start + 10)
        X[i] = Data_Functions.to_categorical(arith_series, n_col = 18)
        y[i] = np.roll(X[i], -1, axis = 0)
    y[:, -1, 1] = 1 # Mark end of sequence
    return X, y

X, y = generate_arithmetic_series(3000)
X_train, X_test, y_train, y_test = Data_Functions.train_test_split(X, y, test_size = 0.5)

rnn = Base_Neural_Network(optimizer = Optimizers.Adam(), loss = Loss_Functions.categorical_cross_entropy, loss_grad = Loss_Functions.categorical_cross_entropy_grad)
rnn.add(RNN(n_units = 10, activation = "tanh", bp_time_steps = 5, input_shape = (10, 18)))
rnn.add(BatchNormalization())
rnn.add(Activation("softmax"))

lstm = Base_Neural_Network(optimizer = Optimizers.Adam(), loss = Loss_Functions.categorical_cross_entropy, loss_grad = Loss_Functions.categorical_cross_entropy_grad)
lstm.add(LSTM(mem_cells = 10, bp_time_steps = 2, input_shape = (10, 18)))
lstm.add(Activation("softmax"))

epochs_RNN = 650
epochs_LSTM = 650

train_error, _ = rnn.fit(X_train, y_train, epochs = epochs_RNN, batch_size = 500)
train_error_lstm, _ = lstm.fit(X_train, y_train, epochs = epochs_LSTM, batch_size = 500)

y_pred = np.argmax(rnn.predict(X_test), axis = 2)
y_pred_lstm = np.argmax(lstm.predict(X_test), axis = 2)
y_test = np.argmax(y_test, axis = 2)

accuracy = np.mean(Data_Functions.accuracy_score(y_test, y_pred))
accuracy_lstm = np.mean(Data_Functions.accuracy_score(y_test, y_pred_lstm))
print("RNN Accuracy:", accuracy)
print("LSTM Accuracy:", accuracy_lstm)

training = plt.plot(range(epochs_RNN), train_error, label = "RNN Training Error")
training_lstm = plt.plot(range(epochs_LSTM), train_error_lstm, label = "LSTM Training Error")
plt.title("Error Plot")
plt.ylabel('Training Error')
plt.xlabel('Iterations')
plt.show()


