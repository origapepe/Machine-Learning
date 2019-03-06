import numpy as np


class NeuralNetwork:
    def __init__(self, i, o):
        self.i = i
        self.o = o

        self.alpha = 0.01
        self.beta = 0.9

        self.weights = []
        self.biases = []
        self.activations = []

        self.vW = []
        self.vB = []

    def add(self, n, act):
        if len(self.weights) == 0:
            self.weights.append(np.random.rand(n, self.i)-0.5)
        else:
            self.weights.append(np.random.rand(n, self.weights[len(self.weights)-1].shape[0])-0.5)

        self.biases.append(np.random.rand(n, 1)-0.5)
        self.vW.append(np.zeros((n, 1)))
        self.vB.append(np.zeros((n, 1)))
        self.activations.append(act)

    def init(self, act):
        self.weights.append(np.random.rand(self.o, self.weights[len(self.weights)-1].shape[0])-0.5)
        self.biases.append(np.random.rand(self.o, 1)-0.5)
        self.activations.append(act)
        self.vW.append(np.zeros((self.o, 1)))
        self.vB.append(np.zeros((self.o, 1)))

    def predict(self, x):
        layer = x
        for i in range(len(self.weights)):
            layer = np.dot(self.weights[i], layer)
            layer = np.add(layer, self.biases[i])
            f = np.vectorize(self.activations[i][0])
            layer = f(layer)
        return layer

    def train(self, x, y):
        layers = [x]
        for i in range(len(self.weights)):
            layers.append(np.dot(self.weights[i], layers[i]))
            layers[i+1] = np.add(layers[i+1], self.biases[i])
            f = np.vectorize(self.activations[i][0])
            layers[i+1] = f(layers[i+1])

        delta = np.multiply(np.subtract(layers[len(layers)-1], y), 2)
        for i in reversed(range(len(self.weights))):
            f = np.vectorize(self.activations[i][1])
            delta = np.multiply(delta, f(layers[i+1]))
            self.vB[i] = self.vB[i]*self.beta + (1-self.beta)*delta
            self.biases[i] = np.subtract(self.biases[i], self.alpha*self.vB[i])
            grad = np.dot(delta, np.transpose(layers[i]))
            self.vW[i] = self.vW[i]*self.beta + (1-self.beta)*grad
            self.weights[i] = np.subtract(self.weights[i], self.alpha*self.vW[i])
            delta = np.dot(np.transpose(self.weights[i]), delta)

    def cost(self, x, y):
        return np.sum(np.power(np.subtract(self.predict(x), y), 2))

