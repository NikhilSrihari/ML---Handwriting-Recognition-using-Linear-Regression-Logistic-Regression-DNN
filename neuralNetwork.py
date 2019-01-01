from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras import optimizers

import numpy as np

input_size = 10
drop_out = 0.1
first_dense_layer_nodes  = 500
second_dense_layer_nodes = 500
third_dense_layer_nodes = 4


class NeuralNetwork():

    def __init__(self, learningRate, momentum , decay):
        self.NN = Sequential()
        (self.NN).add(Dense(first_dense_layer_nodes, input_dim=input_size))
        (self.NN).add(Activation('relu'))
        (self.NN).add(Dropout(drop_out))
        (self.NN).add(Dense(second_dense_layer_nodes))
        (self.NN).add(Activation('relu'))
        (self.NN).add(Dropout(drop_out))
        (self.NN).add(Dense(third_dense_layer_nodes))
        (self.NN).add(Activation('softmax'))
        optimizer = optimizers.SGD( lr=learningRate, decay=decay, momentum=momentum, nesterov=True)
        (self.NN).compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])


    def fit(self, trainingData_features0, trainingData_target0, validationDatasetPercentage, numOfEpochs, modelBatchSize, tbBatchSize, earlyPatience):
        trainingData_features = np.array(trainingData_features0.copy())
        trainingData_target = np.array(trainingData_target0.copy())
        history = (self.NN).fit(trainingData_features, trainingData_target, validation_split=(validationDatasetPercentage/100), 
            epochs=numOfEpochs, batch_size=modelBatchSize, 
            callbacks = [TensorBoard(log_dir='logs', batch_size= tbBatchSize, write_graph= True), EarlyStopping(monitor='val_loss', verbose=1, patience=earlyPatience, mode='min')])


    def predict(self, testingData_features0):
        testingData_features = np.array(testingData_features0.copy())
        return model.predict(testingData_features)