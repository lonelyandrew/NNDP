#!/usr/bin/env python3

import numpy as np


class NeuralNetwork:
    '''A neural network implementation.
    '''

    def __init__(self, layer_size, init_weights=False, init_biases=False):
        '''Initialize the network with the count of layers, and their size.

        Args:
            layer_size: An array-like object which hold the size of neurons
                inside layers respectively.
            init_weights (optional): user-given weights to initialize
            init_biases (optional): user-given biases to initialize
        '''
        self.layer_size = layer_size
        self.nlayer = len(layer_size)

        if init_weights:
            # self.weights = init_weights
            self.weights = [np.ones((y, x))
                            for x, y in zip(layer_size[:-1], layer_size[1:])]
        else:
            self.weights = [np.random.randn(m, n) for m, n in zip(
                layer_size[1:], layer_size[:-1])]

        if init_biases:
            # self.biases = init_biases
            self.biases = [np.ones((y, 1)) for y in layer_size[1:]]
        else:
            self.biases = [np.random.randn(m, 1) for m in layer_size[1:]]

    @staticmethod
    def sigmoid(z):
        '''Compute the sigmoid function value of given value.

        Args:
            z: The argument.

        Return:
            Function value.
        '''
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        '''Compute the derivative of the sigmoid function.
        '''
        return NeuralNetwork.sigmoid(z) * (1 - NeuralNetwork.sigmoid(z))

    def feedforward(self, x):
        '''Process feedforward procedure.

        Args:
            x: The input vector.

        Returns:
            The output vector of the network.
        '''
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, x) + b
            x = self.sigmoid(z)
        return x

    def sgd(self, training_set, batch_size, eta, epoch, test_set=None):
        '''Train with the stochastic gradient descent.

        Args:
            training_set: The training dataset. A List of tuples which has
                two vectors: features x, and the lable y.
            batch_size: The count of training samples in a batch.
            eta: The learning rate.
            epoch: How much turns to train over the whole dataset.
            test_set (optional): The test set to evaluate after training.
        '''
        training_set = list(training_set)
        for i in range(epoch):
            n = len(training_set)
            np.random.shuffle(training_set)
            batches = (training_set[k: k + batch_size]
                       for k in range(0, n, batch_size))
            for batch in batches:
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                nabla_b = [np.zeros(b.shape) for b in self.biases]

                for x, y in batch:
                    bp_w, bp_b = self.back_propa(x, y)
                    nabla_w = [nw + nnw for nw, nnw in zip(nabla_w, bp_w)]
                    nabla_b = [nb + nnb for nb, nnb in zip(nabla_b, bp_b)]

                self.weights = [w - (eta / batch_size) * nw for w, nw in
                                zip(self.weights, nabla_w)]
                self.biases = [b - (eta / batch_size) * nb for b, nb in
                               zip(self.biases, nabla_b)]

            if test_set:
                test_set = list(test_set)
                acc = self.evaluate(test_set)
                print('Epoch {0} finished: {1:.2%}'.format(i+1, acc))
            else:
                print('Epoch {0} finished.'.format(i+1))

    def evaluate(self, test_set):
        '''Evaluate testset.

        Args:
            test_set: the test_set data.

        Returns:
            Return a float accuracy.
        '''
        results = ((np.argmax(self.feedforward(x)), y) for x, y in test_set)
        acc = np.sum([x == y for x, y in results])
        return acc / len(test_set)

    def back_propa(self, x, y):
        '''Get the propagation result from a single example.

        Args:
            x: feature vector of one single training sample.
            y: Corresponsing label.

        Returns:
            A tuple which contains gradient of w and b.
        '''
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        zs = []
        activations = [x]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, x) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
            x = activation

        w_error = activations[-1] - y

        for i in range(1, self.nlayer):
            error = w_error * self.sigmoid_prime(zs[-i])
            nabla_w[-i] = np.dot(error, activations[-i-1].T)
            nabla_b[-i] = error
            w_error = np.dot(self.weights[-i].T, error)

        return (nabla_w, nabla_b)
