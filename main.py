#! /usr/bin/env python3

from neural_network.simple_network.network import NeuralNetwork
from neural_network.data.mnist_loader import load_data_wrapper
# import numpy as np


if __name__ == '__main__':
    print(NeuralNetwork.sigmoid(-1.28 - 0.98))