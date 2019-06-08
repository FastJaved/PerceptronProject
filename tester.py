import numpy as np

def getTestResults(perceptrons, testsData):
    results = {}
    for test_serie_name in testsData.keys():
        results[test_serie_name] = {'predicted':[], 'expected':testsData[test_serie_name]['expected']}
        for perceptron in perceptrons:
            if perceptron.predict(np.array(testsData[test_serie_name]['layer'])) == 1:
                results[test_serie_name]['predicted'].append(perceptron.genre)
    return results