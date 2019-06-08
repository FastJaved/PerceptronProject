import json
import time

import pandas as pd

from OrderedSet import OrderedSet
from keywords import extractKeyWords
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
    print("#### Starting ####")
    nb_series = 2000

    with open('series_' + str(nb_series) + '.json', encoding="utf8") as json_file:
        data = json.load(json_file)

    series = pd.DataFrame(data["series"])

    #print(tabulate(series[:10], headers='keys', tablefmt='psql'))
    #print(tabulate(target[:10], headers='keys', tablefmt='psql'))

    genres = ['Science Fiction', 'Fantasy', 'Drama', 'Action', 'Adventure', 'Comedy', 'Crime', 'Mystery', 'Family', 'War & Politics', 'Horror', 'Romance', 'Documentary', 'Reality', 'Western', 'Kids', 'News', 'Music', 'Talk', 'Soap', 'History']

    allWords = getAllWords(series['overview'])

    with open('layer_order_' + str(nb_series) + '.json', 'w') as layer_order:
        json.dump(allWords, layer_order)

    perceptrons = []

    nbTests = 20

    for i in genres[:3]:
        print("### Genre : " + i)
        perceptron = Perceptron(len(allWords))
        perceptron.genre = i
        perceptron.nbSeries = nb_series
        perceptrons.append(perceptron)

        inputs = []
        outputs = []

        print("#### Init data ####")

        for x in range(nbTests,len(series)):
            layer = getLayer(series['overview'][x], allWords)

            inputs.append(layer)

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
        tests[series["name"][i]] = getLayer(series['overview'][i], allWords)

    results = {}
    for test_serie_name in tests.keys():
        results[test_serie_name] = []
        for perceptron in perceptrons:
            if perceptron.predict(np.array(tests[test_serie_name])) == 1:
                results[test_serie_name].append(perceptron.genre)
    print("results")
    print(results)












