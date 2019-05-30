import json
import multiprocessing

import pandas as pd
from tabulate import tabulate
from lucas import extractKeyWords
import numpy as np
from perceptronClass import Perceptron


def getAllWords(series):
    allWords = set()
    for serie in series:
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

    with open('series_20.json', encoding="utf8") as json_file:
        data = json.load(json_file)

    series = pd.DataFrame(data["series"])

    target = pd.DataFrame()
    target["id"] = series["id"]
    target["genres"] = series["genres"]
    #print(tabulate(series[:10], headers='keys', tablefmt='psql'))
    #print(tabulate(target[:10], headers='keys', tablefmt='psql'))

    with open('dictLucas.json', encoding="utf8") as json_file:
        dict = json.load(json_file)

    synopsis = series["overview"][2]
    genre = target["genres"][2]

    synopsis = extractKeyWords(synopsis)

    genres = dict.keys()
    allWords = getAllWords(series['overview'])
    lr = 10 # learning rate
    steps = 10
    perceptron = Perceptron(len(allWords)) # initialize a perceptron
    perceptron.genre = 'Drama'
    perceptron.nbThreads = 4

    inputs = []
    outputs = []
    tests = []

    print("#### Init data ####")

    for x in range(len(series)) :
        layer = getLayer(series['overview'][x], allWords)

        inputs.append(layer)
        tests.append(layer)

        if (perceptron.genre in series['genres'][x]):
            outputs.append(1)
        else :
            outputs.append(0)

    ts_input = np.array(inputs)
    ts_output = np.array(outputs).T
    testing_data = np.array(tests)

    print(ts_output)

    print("#### Training ####")

    #perceptron.train(ts_input, ts_output, steps, lr) # train the perceptron

    jobs = []
    nbThreads = 5
    print("PART : " +  str(steps//nbThreads))
    for k in range(nbThreads):
        p = multiprocessing.Process(target=perceptron.train,
                                    args=(ts_input, ts_output, steps//nbThreads, lr))
        jobs.append(p)

    for p in jobs:
        p.start()
    for p in jobs:
        p.join()

    results = []
    for x in (range(len(testing_data))):
        run = testing_data[x]
        trial = perceptron.results(run)
        results.append(trial.tolist())
    print("results")
    print(results)
    print(np.ravel(np.rint(results)))













