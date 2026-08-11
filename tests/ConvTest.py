import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.Base_Neural_Network import Base_Neural_Network
from utils import Optimizers
from utils import Loss_Functions
from utils.Layers import Dense, Activation, Dropout, BatchNormalization, Flatten, Conv2D
from utils.Data_Functions import Data_Functions 

from sklearn import datasets
import matplotlib.pyplot as plt

data = datasets.load_digits()
X = data.data
y = data.target

y = Data_Functions.to_categorical(y.astype("int"))

X_train, X_test, y_train, y_test = Data_Functions.train_test_split(X, y, test_size = 0.4)

X_train = X_train.reshape(-1, 1, 8, 8)
X_test = X_test.reshape(-1, 1, 8, 8)

conv_model = Base_Neural_Network(Optimizers.Adam(), loss = Loss_Functions.categorical_cross_entropy, loss_grad = Loss_Functions.categorical_cross_entropy_grad)
conv_model.add(Conv2D(filters = 16, filter_shape = (3, 3), input_shape = (1, 8, 8), padding_type = "same", stride = 1))
conv_model.add(Activation("relu"))
conv_model.add(Dropout(probability = 0.25))
conv_model.add(BatchNormalization())
conv_model.add(Conv2D(filters = 32, filter_shape = (3, 3), padding_type = "same", stride = 1))
conv_model.add(Activation("relu"))
conv_model.add(Dropout(probability = 0.25))
conv_model.add(BatchNormalization())
conv_model.add(Flatten())
conv_model.add(Dense(n_units = 256))
conv_model.add(Activation("relu"))
conv_model.add(Dropout(probability = 0.4))
conv_model.add(BatchNormalization())
conv_model.add(Dense(n_units = 10))
conv_model.add(Activation("softmax"))

training_loss, val_loss = conv_model.fit(X_train, y_train, epochs = 50, batch_size = 256)
training_plot, = plt.plot(range(len(training_loss)), training_loss, label = "Training loss")
val_plot, = plt.plot(range(len(val_loss)), val_loss, label = "Validation loss")

plt.legend(handles = [training_plot, val_plot])
plt.title("Loss Plot")
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.show()