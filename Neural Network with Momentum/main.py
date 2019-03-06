import numpy as np
import matplotlib.pyplot as plt
from nn import NeuralNetwork

print("Do it again and do it right!")

np.random.seed(0)

data = []
N = 1000

for i in range(N):
    x = i*np.pi*2/N
    data.append([[[x]], [np.sin(x)]])


def tanh(x):
    return np.tanh(x)


def Dtanh(y):
    return 1-y*y


def linear(x):
    return x


def Dlinear(y):
    return 1


nn = NeuralNetwork(1, 1)

nn.add(5, [tanh, Dtanh])
nn.add(5, [tanh, Dtanh])
nn.init([linear, Dlinear])

n = 25000
cost = []

for i in range(n):
    index = np.random.randint(0, N)
    x = data[index][0]
    y = data[index][1]
    c = nn.cost(x, y)
    cost.append(c)
    nn.train(x, y)

    if i % 100 == 0:
        print("Iter " + str(i) + "  Cost " + str(c))

nnG = []
sin = []
for i in range(N):
    x = i*np.pi*2/N
    nnG.append(nn.predict(x)[0])
    sin.append(np.sin(x))

c = 0
for i in range(len(nnG)):
    c += nnG[i]/len(nnG)

print(c)
print("____")
print(nn.predict([[0]]))
print(nn.predict([[np.pi/2]]))

plt.plot(np.arange(0.0, n), cost)
plt.ylabel("Cost")
plt.xlabel("Time")
plt.show()
plt.plot(np.linspace(0.0, 2*np.pi, N), nnG, 'b')
plt.plot(np.linspace(0.0, 2*np.pi, N), sin, 'r')
plt.show()
