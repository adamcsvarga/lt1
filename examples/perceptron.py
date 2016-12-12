#!/usr/bin/env python3
# By Jon Dehdari, 2016
# Comments by Adam Varga
"""Implements simple perceptron training and classification algorithm.

Can be used for various classification tasks. To run the example script type:
python3 -i perceptron.py

"""

## import stuff for IO, arrays and plotting
import sys
import numpy as np
import matplotlib.pyplot as plt

# class labels
labels = [0, 1] 
# no. of samples to use
#num_samples = 30
num_samples = 200
# set random seed for reproducability
np.random.seed(seed=3)
# learning rate AKA eta AKA epsilon
learning_rate = 1.0
# set split ratio b/w test and traon
test_percentage = 10
# calculate train and test size based on ratio
test_size = num_samples // test_percentage
train_size = num_samples - test_size


class Perceptron(list):
    """Implements a simple perceptron.
    
    Attributes:
        params: weight matrix to train
        bias: bias term determining the offset
        learning_rate: what learning rate to use during training
        
    Usage example:

    >>> fig = plt.figure()
    >>> subplot = fig.add_subplot(1,1,1, xlim=(-5,5), ylim=(-5,5))
    >>> train_set = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
    >>> test_set  = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
    >>> p = Perceptron(2)
    >>> p.train(train_set, test_set, status=10, subplot=subplot, fig=fig)
    """

    def __init__(self, num_inputs, learning_rate=learning_rate):
        """Class constructor.
        
        Args:
            num_inputs: input size; type: int
            learning_rate: learning rate (use global learning rate 
            if not specified); type: float
        """
        # random initialization of parameters 
        self.params = np.random.normal(size=num_inputs)
        self.bias   = np.random.normal()
        self.learning_rate = learning_rate

    def __repr__(self):
        """For priniting perceptron parameters.
        
        Returns:
            string containing perceptron's parameters and bias
        """
        return str(np.concatenate((self.params, [self.bias])))


    def error(self, guess, correct):
        """Measure error between predictions and true labels.
       
        Args:
            guess: predicted label
            correct: true label
        
        Returns:
            error between predited and true labels (float)
            
        """
        return correct - guess


    def activate(self, val, fun='step'):
        """Implementation of the perceptron's activation function
        
        Args:
            val: perceptron's input values
            fun: shape of activation function ('step' or 'tanh')
            
        Returns:
            output of the activation function given an input value
        """
        # Simple step activation: return 1 if input is positive
        if fun == 'step':
            if val >= 0:
                return 1
            else:
                return 0
        # Tanh activation
        elif fun == 'tanh':
            return np.tanh(val)


    def predict(self, x):
        """Get perceptron's output for a given vector.
        
        Args:
            x: vector for which the output should be acquired
            
        Returns:
            Perceptron's output for x as input
        """
        #if 0 > np.tanh(np.dot(x, self.params) + self.bias):
        return self.activate(np.dot(x, self.params) + self.bias)  # Multiply instance with parameter matrix, add bias, then calculate perceptron's activation.


    def predict_set(self, test_set):
        """Calculate accuracy on test set.
        
        Args:
            test_set: test vectors to calculate accuracy on
            
        Returns:
            Classification accuracy on test set.
        """
        # Start with 0 errors
        errors = 0
        # Iterate through vectors in test set
        for x, y in test_set:
            # Get output for each test instance
            out = self.predict(x)
            # If prediction doesn't match true class, increment no. of errors.
            if out != y:
                errors +=1 
        # Calculate classification accuracy
        return 1 - errors / len(test_set)


    def decision_boundary(self):
        """ Returns two points, along which the decision boundary for a binary classifier lies. """
        return ((0, -self.bias / self.params[1]), (-self.bias / self.params[0], 0))


    def train(self, train_set, dev_set, status=100, epochs=10, subplot=None, fig=None):
        """Implementation of the training process.
        
        Args:
            train_set: training vectors
            dev_set: development vectors
            status: how often to monitor dev accuracy (as the no. of training samples)
            epochs: number of training epochs
            subplot: which subplot to plot in
            fig: which figure to use when plotting
        """
        # Get initial accuracy on dev set
        print("Starting dev set accuracy: %g; Line at %s; Params=%s" % (self.predict_set(dev_set), str(self.decision_boundary()), str(self)), file=sys.stderr)
        # Plot initial decision boundary.
        subplot.plot([0, -self.bias/self.params[1]], [-self.bias/self.params[0], 0], '-', color='lightgreen', linewidth=1)
        fig.canvas.draw()

        # Perform a given number of training epochs
        iterations = 0
        for epoch in range(epochs):
            np.random.shuffle(train_set)  # Randomize training set to make sure it's not ordered (that might lead to the training not being able to converge(?))
            for x, y in train_set:
                iterations += 1
                # Get perceptron output and corresponding error.
                out = self.predict(x)
                error = self.error(out, y)
                # If error is non-zero, adjust bias and parameters depending on learning rate & error
                if error != 0:
                    #print("out=", out, "; y=", y, "; error=", error, file=sys.stderr)
                    self.bias += self.learning_rate * error # Change bias according to learning rate & error
                    for i in range(len(x)):
                        self.params[i] += self.learning_rate * error * x[i]  # Adjust parameters depending on learning rate & error & input vector
                # Monitor accuracy and plot decision boundary after every n sample (where n=status)
                if iterations % status == 0:
                    print("Dev set accuracy: %g; Line at %s; Params=%s" % (self.predict_set(dev_set), str(self.decision_boundary()), str(self)), file=sys.stderr)
                    subplot.plot([0, -self.bias/self.params[1]], [-self.bias/self.params[0], 0], '-', color='lightgreen', linewidth=1)
                    fig.canvas.draw()
            self.learning_rate *= 0.9  # Adaptive learning rate adjustment (start fast, then gradually slow down)
        # Plot final decision boundary.
        subplot.plot([0, -self.bias/self.params[1]], [-self.bias/self.params[0], 0], 'g-x', linewidth=5)
        fig.canvas.draw()


def main():
    """Main function.
    
    Creates example data sets then performs training while monitoring the
    accuracy and plotting intermediate and final decision boundaries.
    """
    
    import doctest
    doctest.testmod()  # Test the code snippet in the docstring

    # Create "blue" and "red" training (x) and testing (y) instances.
    # The instances are sampled from normal distributions.
    # "Blue" has zero mean, while "red"'s mean is at 2.
    # The test instances are sampled from distributions with 0.5 standard
    # deviation, while the training isntances' standard deviation is 1.0 (default)
    x_blue = np.random.normal(loc=0, size=num_samples)
    y_blue = np.random.normal(loc=0, scale=0.5, size=num_samples)
    x_red  = np.random.normal(loc=2, size=num_samples)
    y_red  = np.random.normal(loc=2, scale=0.5, size=num_samples)
    
    # Create red and blue classes, labeling them with the appropriate class label
    # Put these together in the data list
    data =  list(zip(zip(x_blue, y_blue), [labels[0]] * num_samples))
    data += zip(zip(x_red,  y_red),  [labels[1]] * num_samples)
    np.random.shuffle(data)  # Shuffle data to avoid class bias when splitting to test and train sets
    
    # Split to train and test
    train_set = data[:train_size]
    test_set  = data[train_size:]
    
    # Matplotlib craziness to be able to update a plot
    plt.ion()
    fig = plt.figure()
    subplot = fig.add_subplot(1,1,1, xlim=(-5,5), ylim=(-5,5))
    subplot.grid()
    subplot.plot(x_blue, y_blue, linestyle='None', marker='o', color='blue')
    subplot.plot(x_red,  y_red,  linestyle='None', marker='o', color='red')

    # Initialize perceptron with two-dimensional input
    p = Perceptron(2)
    # Perform training
    p.train(train_set, test_set, subplot=subplot, fig=fig)


if __name__ == '__main__':
    main()
