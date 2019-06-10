import json
import time

import pandas as pd

from OrderedSet import OrderedSet
from keywords import extractKeyWords
import numpy as np
from perceptron import Perceptron
from tester import getTestResults


def getAllWords(series):
    allWords = OrderedSet()  # So words are always in the same order
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
    print("#### Starting ####")
    nb_series = 10000 #TODO make it automatic

    with open('series_OMDB_plot.json', encoding="utf8") as json_file:
        data = json.load(json_file)

    series = pd.DataFrame(data["series"])

    # print(tabulate(series[:10], headers='keys', tablefmt='psql'))
    # print(tabulate(target[:10], headers='keys', tablefmt='psql'))

    genres = ['Science Fiction', 'Fantasy', 'Drama', 'Action', 'Adventure', 'Comedy', 'Crime', 'Mystery', 'Family',
              'War & Politics', 'Horror', 'Romance', 'Documentary', 'Reality', 'Western', 'Kids', 'News', 'Music',
              'Talk', 'Soap', 'History'] #TODO automatic list

    allWords = getAllWords(series['overview'])

    with open('layer_order_' + str(nb_series) + '.json', 'w') as layer_order:
        json.dump(allWords, layer_order)

    perceptrons = []

    nbTests = 1500

    inputs = []
    for x in range(nbTests, len(series)):
        layer = getLayer(series['overview'][x], allWords)
        inputs.append(layer)

    for i in genres:
        print("### Genre : " + i)
        perceptron = Perceptron(len(allWords))
        perceptron.genre = i
        perceptron.nbSeries = nb_series
        perceptrons.append(perceptron)

        outputs = []

        for x in range(nbTests, len(series)):
            if (perceptron.genre in series['genres'][x]):
                outputs.append(1)
            else:
                outputs.append(0)

        ts_input = np.array(inputs)
        ts_output = np.array(outputs).T

        print("#### Training ####")
        start_time = time.time()
        perceptron.train(ts_input, ts_output)  # train the perceptron
        print("--- %s seconds ---" % (time.time() - start_time))

    tests = {}
    for i in range(nbTests):
        tests[series["name"][i]] = {'layer': getLayer(series['overview'][i], allWords), 'expected': series["genres"][i]}

    with open('tests_' + str(nbTests) + '_out_of_' + str(nb_series) + '.json', 'w') as tests_file:
        json.dump(tests, tests_file)

    results = getTestResults(perceptrons, tests)
    print(results)
