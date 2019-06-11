import numpy as np
import json

class Perceptron(object):
    def __init__(self, no_of_inputs, steps=200, learning_rate=0.01):
        self.steps = steps
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
        self.genre = ''
        self.nbSeries = 0
        self.score = 0
        self.scoreWhenTrue = 0
        self.scoreWhenFalse = 0

    def predict(self, inputs):
        summation = self.getSummation(inputs)
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def getSummation(self, inputs):
        return np.dot(inputs, self.weights[1:]) + self.weights[0]

    def getPercentage(self, n, numberOfIteration):
        return n * 100 / numberOfIteration

    def updateWeights(self, label, prediction, inputs):
        self.weights[1:] += self.learning_rate * (label - prediction) * inputs
        self.weights[0] += self.learning_rate * (label - prediction)

    def saveWeights(self):
        try:
            with open('syn_weights_' + str(self.nbSeries) + '.json', 'r') as f:
                data = json.load(f)
        except:
            data = []
            dict = {str(self.genre): self.weights.tolist()}
            data.append(dict)
            with open('syn_weights_' + str(self.nbSeries) + '.json', 'w') as outfile:
                json.dump(data, outfile)
        else :
            dict = {str(self.genre): self.weights.tolist()}
            data.append(dict)
            with open('syn_weights_' + str(self.nbSeries) + '.json', 'w') as outfile:
                json.dump(data, outfile)

    def train(self, training_inputs, labels):
        count = 0
        for _ in range(self.steps):
            for inputs, label in zip(training_inputs, labels):
                count += 1
                print(str(self.getPercentage(count, self.steps * len(training_inputs))) + '%')
                prediction = self.predict(inputs)
                self.updateWeights(label, prediction, inputs)
        self.saveWeights()
