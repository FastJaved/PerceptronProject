import numpy as np
import json

from perceptron import Perceptron
from tester import getTestResults
import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def getPercentage(nbSuccess, nbTot):
    return nbSuccess * 100 / nbTot


def getScore(expectedResults, actualResults):
    score = 0

    for expected, actual in zip(expectedResults, actualResults):
        if expected == actual:
            score += 1

    return getPercentage(score, len(expectedResults))


def plotScores(perceptrons):
    x = []
    y = []

    for perceptron in perceptrons:
        x.append(perceptron.genre)
        y.append(perceptron.score)

    y, x = zip(*sorted(zip(y, x)))

    y_pos = np.arange(len(x))

    plt.bar(range(len(x)), y, align='center', color='#7ed6df')

    plt.xticks(y_pos, x, fontsize=5, rotation=30)
    plt.ylabel('Score (in %)', weight='bold', size='large')
    plt.xlabel('Genre', weight='bold', size='large')
    plt.title('Score prediction per genre')

    plt.ylim(ymin=np.amin(y) - 5, ymax=100)
    print("Average is " + str(np.average(y)))
    plt.grid(True, which="both", linestyle='--')
    plt.show()


def plotScoreDetails(perceptrons):
    x = []

    y0 = []
    y1 = []

    for perceptron in perceptrons:
        x.append(perceptron.genre)
        y0.append(perceptron.scoreWhenFalse)
        y1.append(perceptron.scoreWhenTrue)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(len(x))
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, y0, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Score when false')

    rects2 = plt.bar(index + bar_width, y1, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Score when true')

    plt.xlabel('Genres')
    plt.ylabel('Scores %')
    plt.title('Score details for each froup')
    plt.xticks(index + bar_width, x, fontsize=5, rotation=30)
    plt.legend()
    plt.grid(True, which="both", linestyle='--')

    print("Average when true " + str(np.average(y1)))
    print("Average when false " + str(np.average(y0)))
    plt.tight_layout()
    plt.show()


nb_series = 10000

###### LOAD LAYER_ORDER #######
with open('layer_order_' + str(nb_series) + '.json', encoding="utf8") as layer_order:
    layer_order = json.load(layer_order)

allWords = layer_order

###### LOAD PERCEPTRONS ######
with open('syn_weights_' + str(nb_series) + '.json', encoding="utf8") as json_file:
    data = json.load(json_file)

perceptrons = []

for k in data:
    genre = list(k.keys())[0]
    perceptron = Perceptron(len(allWords))
    perceptron.genre = genre
    perceptron.nbSeries = nb_series
    perceptron.weights = np.array(list(k.values())[0])
    perceptrons.append(perceptron)

###### LOAD TEST_DATAS ######
with open('tests_1500_out_of_10000.json', encoding="utf8") as test_file:
    test_datas = json.load(test_file)

results = getTestResults(perceptrons, test_datas)
print(results)

#perceptrons.sort(key=lambda x: x.nbSeries, reverse=True)

for perceptron in perceptrons:
    exp = []
    pred = []
    for res in results:
        currentExpected = results[res]['expected']
        currentPredicted = results[res]['predicted']

        exp.append(perceptron.genre in currentExpected)
        pred.append(perceptron.genre in currentPredicted)

    perceptron.score = getScore(np.array(exp), np.array(pred))

for perceptron in perceptrons:
    exp = []
    pred = []

    exp1 = []
    pred1 = []

    exp0 = []
    pred0 = []

    nb = 0
    for res in results:
        currentExpected = results[res]['expected']
        currentPredicted = results[res]['predicted']

        exp.append(perceptron.genre in currentExpected)
        pred.append(perceptron.genre in currentPredicted)

        if perceptron.genre in currentExpected:
            exp1.append(perceptron.genre in currentExpected)
            pred1.append(perceptron.genre in currentPredicted)
            nb += 1
        else:
            exp0.append(perceptron.genre in currentExpected)
            pred0.append(perceptron.genre in currentPredicted)

    print("nb for " + perceptron.genre + ": " + str(nb))
    perceptron.score = getScore(np.array(exp), np.array(pred))
    perceptron.scoreWhenTrue = getScore(np.array(exp1), np.array(pred1))
    perceptron.scoreWhenFalse = getScore(np.array(exp0), np.array(pred0))

plotScores(perceptrons)
plotScoreDetails(perceptrons)

# expectedResults = np.array([0, 0, 1, 0, 1, 1])
# actualResults = np.array([1, 1, 1, 1, 0, 1])
# print("{:.2f}".format(getScore(expectedResults, actualResults)) + " %")
