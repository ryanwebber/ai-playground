import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

class Model:
    def __init__(self, model, input_size):
        self.model = model
        self.input_size = input_size

    def predict(self, obs):
        return np.argmax(self.model.predict(obs.reshape(-1, self.input_size, 1))[0])

def _create_model(input_size, output_size, learning_rate=1e-3):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, output_size, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train(training_data, epochs=3, run_id="cartpole-v0-training"):

    # The length of the observations returned from the sim
    input_size = len(training_data[0][0])

    # The length of the one-hot encoding (will become output layer size)
    output_size = len(training_data[0][1])

    X = np.array([i[0] for i in training_data]).reshape(-1, input_size, 1)
    y = [i[1] for i in training_data]

    model = _create_model(input_size, output_size)

    model.fit({'input': X}, {'targets': y}, n_epoch=epochs, snapshot_step=500, show_metric=True, run_id=run_id)

    return Model(model, input_size)

