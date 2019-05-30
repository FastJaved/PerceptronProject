import numpy as np
import json

class Perceptron():
    def __init__(self, size):
        self.syn_weights = np.random.rand(size,1)
        self.genre = ''
        self.numberOfWords = size

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

    def getPercentage(self, n, numberOfIteration):
        return n * 100 / numberOfIteration

    def train(self, inputs, real_outputs, its, lr):
        delta_weights = np.zeros((self.numberOfWords, len(inputs)))
        count = 0
        for iteration in (range(its)):
            # Forward Pass
            z = np.dot(inputs, self.syn_weights)
            activation = self.sigmoid(z)
            # Backward Pass
            for i in range(len(inputs)):
                count += 1
                print(str(self.getPercentage(count, its * len(inputs))) + '%')
                cost = (activation[i] - real_outputs[i]) ** 2
                cost_prime = 2 * (activation[i] - real_outputs[i])
                for n in range(self.numberOfWords):
                    delta_weights[n][i] = cost_prime * inputs[i][n] * self.sigmoid_deriv(z[i])
            delta_avg = np.array([np.average(delta_weights, axis=1)]).T
            self.syn_weights = self.syn_weights - delta_avg * lr
        with open('syn_weights_' + str(self.genre) + '.json', 'w') as outfile:
            json.dump(self.syn_weights, outfile)


    def results(self, inputs):
        return self.sigmoid(np.dot(inputs, self.syn_weights))

