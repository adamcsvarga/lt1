# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:24:31 2017

Simple feedforward neural network example.

@author: Adam Varga
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np


## output labels (0=blue, 1=red)
labels = [0, 1]
## number of samples to generate
num_samples = 200
## set random seed for repoducability
np.random.seed(seed=3)
## set learning rate
learning_rate = 0.03
## train to test ratio setup and calculation
test_percentage = 10
test_size = num_samples // test_percentage
train_size = num_samples - test_size

def build_nw(num_hidden_layers, hidden_layer_dim, activation="relu"):
    """Builds up a feed-forward neural network.
    
    Args:
        num_hidden_layers: number of hidden layers
        hidden_layer_dim: neurons per hidden layer (only layers with same
        dimensions are possible)
        activation: hidden layer activation function
    
    Returns:
        compiled neural network
    """
    
    #model = Sequential([Dense(32, input_dim=2), Activation('relu'), Dense(10), Activation('softmax')])
    model = Sequential()
    
    ## first hidden layer
    model.add(Dense(output_dim=hidden_layer_dim, input_dim=2))
    model.add(Activation(activation))
    
    ## additional hidden layers
    for i in range(1, num_hidden_layers):
        model.add(Dense(output_dim=hidden_layer_dim))
        model.add(Activation(activation))
    ## output layer
    model.add(Dense(output_dim=2))
    model.add(Activation("softmax"))

    ## use sparse categorical crossentropy to make keras accept the format
    ## of the labels array
    model.compile(loss='sparse_categorical_crossentropy', \
                  optimizer=Adam(lr=learning_rate))
    
    return model
    
def train_nw(model, train_set, num_epoch=5, batch_size=32):
    """Neural network training on training data.
    
    Args:
        model: keras NN model
        train_set: list of training instances
        num_epoch: number of training epochs
        batch_size: size of batch
        
    Returns:
        trained model
    
    """
    
    ## get list of training instances
    X = [x[0] for x in train_set ]
    
    ## get list of labels
    Y = [y[1] for y in train_set]
    ## do training with early stopping
    model.fit(X, Y, nb_epoch=num_epoch, batch_size=batch_size)
    
    return model
    
def test_nw(model, test_set, batch_size=32):
    """Use trained network to predict class labels and calculate accuracy
    
    Args:
        model: keras NN model (trained)
        test_set: list of test instances
        batch_size: size of batch
        
    Returns:
        prediction accuracy
    
    """
    
    ## get lists of test instances and labels
    X = [x[0] for x in test_set]
    Y = [y[1] for y in test_set]

    Y_pred = model.predict(X, batch_size=batch_size)
    
    ## calculate accuracy
    correct = 0
    for i in range(0, len(Y_pred)):
        if Y_pred[i][0] >= Y_pred[i][1]:
            if Y[i] == labels[0]:
                correct += 1
        else:
            if Y[i] == labels[1]:
                correct += 1
                
    return correct / len(Y)
    
if __name__ == '__main__':
    
    ## generate data
    x_blue_1 = np.random.normal(loc=4, size=num_samples)
    y_blue_1 = np.random.normal(loc=0, size=num_samples)
    x_blue_2 = np.random.normal(loc=-4, size=num_samples)
    y_blue_2 = np.random.normal(loc=0, size=num_samples)
    x_blue_3 = np.random.normal(loc=0, size=num_samples)
    y_blue_3 = np.random.normal(loc=-4, size=num_samples)
    x_red = np.random.normal(loc=0, size=num_samples)
    y_red = np.random.normal(loc=0, size=num_samples)
    
    x_blue = np.concatenate((x_blue_1, x_blue_2, x_blue_3))
    y_blue = np.concatenate((y_blue_1, y_blue_2, y_blue_3))
    
    ## Create red and blue classes, labeling them with the appropriate class label
    ## Put these together in the data list
    data =  list(zip(zip(x_blue, y_blue), [labels[0]] * num_samples))
    data += zip(zip(x_red,  y_red),  [labels[1]] * num_samples)
    np.random.shuffle(data)  # Shuffle data to avoid class bias when splitting to test and train sets
    
    ## Split to train and test
    train_set = data[:train_size]
    test_set  = data[train_size:]

    
    ## Initialize network
    m = build_nw(1, 2)
    ## Train network
    m = train_nw(m, train_set, num_epoch=500)
    ## Predict labels
    accuracy = test_nw(m, test_set)
    print("Accuracy: " + str(accuracy))
    
    
    
    