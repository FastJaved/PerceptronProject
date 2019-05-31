import json
import time

import pandas as pd

from OrderedSet import OrderedSet
from lucas import extractKeyWords
import numpy as np
from perceptron import Perceptron


def getAllWords(series):
    allWords = OrderedSet() #So words are always in the same order
    count = 0
    for serie in series:
        count += 1
        words = extractKeyWords(serie)
        for word in words.split():
            allWords.add(word)
    allWords = list(allWords)

    return allWords


def getLayer(overview, allWords):
    overview = extractKeyWords(overview)
    layer = [0] * len(allWords)
    for word in overview.split():
        try:
            layer[allWords.index(word)] = 1
        except:
            pass
    return layer

if __name__ == "__main__":

    with open('series_2000.json', encoding="utf8") as json_file:
        data = json.load(json_file)

    series = pd.DataFrame(data["series"])

    #print(tabulate(series[:10], headers='keys', tablefmt='psql'))
    #print(tabulate(target[:10], headers='keys', tablefmt='psql'))

    with open('dictLucas.json', encoding="utf8") as json_file:
        dict = json.load(json_file)

    genres = dict.keys()
    allWords = getAllWords(series['overview'])

    perceptron = Perceptron(len(allWords))
    perceptron.genre = 'Drama'

    inputs = []
    outputs = []
    tests = []

    print("#### Init data ####")

    for x in range(len(series)):
        layer = getLayer(series['overview'][x], allWords)

        inputs.append(layer)

        if (perceptron.genre in series['genres'][x]):
            outputs.append(1)
        else:
            outputs.append(0)

    tests.append(getLayer(series['overview'][0], allWords))
    tests.append(getLayer(series['overview'][1], allWords))
    tests.append(getLayer(series['overview'][2], allWords))

    ts_input = np.array(inputs)
    ts_output = np.array(outputs).T
    testing_data = np.array(tests)

    print(ts_output)

    print("#### Training ####")
    start_time = time.time()
    perceptron.train(ts_input, ts_output)  # train the perceptron
    print("--- %s seconds ---" % (time.time() - start_time))

    results = []
    for x in (range(len(testing_data))):
        trial = perceptron.getSummation(testing_data[x])
        results.append(trial)
    print("results")
    print(results)













