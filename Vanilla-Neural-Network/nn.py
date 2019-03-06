import numpy as np


class NeuralNetwork:
    def __init__(self, i, o):
        self.i = i
        self.o = o

        self.alpha = 0.03

        self.weights = []
        self.biases = []
        self.activations = []

    def add(self, n, act):
        if len(self.weights) == 0:
            self.weights.append(np.random.rand(n, self.i)-0.5)
        else:
            self.weights.append(np.random.rand(n, self.weights[len(self.weights)-1].shape[0])-0.5)

        self.biases.append(np.random.rand(n, 1)-0.5)
        self.activations.append(act)

    def init(self, act):
        self.weights.append(np.random.rand(self.o, self.weights[len(self.weights)-1].shape[0])-0.5)
        self.biases.append(np.random.rand(self.o, 1)-0.5)
        self.activations.append(act)

    def predict(self, x):
        layer = x
        for i in range(len(self.weights)):
            layer = np.dot(self.weights[i], layer)
            layer = np.add(layer, self.biases[i])
            f = np.vectorize(self.activations[i][0])
            layer = f(layer)
        return layer

    def train(self, x, y):
        layers = []
        layers.append(x)
        for i in range(len(self.weights)):
            layers.append(np.dot(self.weights[i], layers[i]))
            layers[i+1] = np.add(layers[i+1], self.biases[i])
            f = np.vectorize(self.activations[i][0])
            layers[i+1] = f(layers[i+1])

        delta = np.multiply(np.subtract(layers[len(layers)-1], y), 2)
        for i in reversed(range(len(self.weights))):
            f = np.vectorize(self.activations[i][1])
            delta = np.multiply(delta, f(layers[i+1]))
            grad = np.multiply(delta, self.alpha)
            self.biases[i] = np.subtract(self.biases[i], grad)
            grad = np.dot(grad, np.transpose(layers[i]))
            self.weights[i] = np.subtract(self.weights[i], grad)
            delta = np.dot(np.transpose(self.weights[i]), delta)

    def cost(self, x, y):
        return np.sum(np.power(np.subtract(self.predict(x), y), 2))

