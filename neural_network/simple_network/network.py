#!/usr/bin/env python3

import numpy as np


class NeuralNetwork:
    '''A neural network implementation.
    '''

    def __init__(self, layer_size):
        '''Initialize the network with the count of layers, and their size.

        Args:
            layer_size: An array-like object which hold the size of neurons
                inside layers respectively.

        Raises:
            ValueError: If the layer_size has uneffective elements.
        '''
        self.layer_size = layer_size
        self.nlayer = len(layer_size)
        self.weights = [np.random.randn(m, n) for m, n in zip(
            layer_size[1:], layer_size[:-1])]
        self.biaes = [np.random.randn(m, 1) for m in layer_size[1:]]

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
        return NeuralNetwork.sigmoid(z) * (1 - NeuralNetwork.sigmoid(z))

    def feedforward(self, x):
        '''Process feedforward procedure.

        Args:
            x: The input vector.

        Returns:
            The output vector of the network.
        '''
        x = np.array(x)
        for w, b in zip(self.weights, self.biaes):
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
        for i in range(epoch):
            n = len(training_set)
            np.random.shuffle(training_set)
            batches = (training_set[k: k + batch_size]
                       for k in range(0, n, batch_size))
            for batch in batches:
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                nabla_b = [np.zeros(b.shape) for b in self.biaes]

                for x, y in batch:
                    bp_w, bp_b = self.back_propa(x, y)
                    nabla_w = [nw + nnw for nw, nnw in zip(nabla_w, bp_w)]
                    nabla_b = [nb + nnb for nb, nnb in zip(nabla_b, bp_b)]

                self.weights = [w - (eta / batch_size) * nw for w, nw in
                                zip(self.weights, nabla_w)]
                self.biaes = [b - (eta / batch_size) * nb for b, nb in
                              zip(self.biaes, nabla_b)]

            if test_set:
                acc = self.evaluate(test_set)
                print('Epoch {0} finished: {1:%.5f}'.format(i, acc))
            else:
                print('Epoch {0} finished.'.format(i))

    def evaluate(self, test_set):
        '''Evaluate testset.

        Args:
            test_set: the test_set data.

        Returns:
            Return a float accuracy.
        '''
        results = ((self.feedforward(x), y) for x, y in test_set)
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
        nabla_b = [np.zeros(b.shape) for b in self.biaes]
        zs = []
        activations = [x]

        for w, b in zip(self.weights, self.biaes):
            z = np.dot(w, x) + b
            zs.append(z)
            activations.append(self.sigmoid(z))

        w_error = activations[-1] - y

        for i in range(1, self.nlayer):
            error = w_error * self.sigmoid_prime(z[-i])
            nabla_w[-i] = activations[-i-1] * error
            nabla_b[-i] = error
            w_error = np.dot(self.weights[-i].T, error)

        return (nabla_w, nabla_b)


if __name__ == '__main__':
    network = NeuralNetwork([3, 2, 1])
    x = network.feedforward([1, 0, 1])
    print(x)
