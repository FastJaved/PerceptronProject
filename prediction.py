import numpy as np

def getPercentage(nbSuccess, nbTot):
    return nbSuccess * 100 / nbTot


def getScore(expectedResults, actualResults):
    score = 0

    for expected, actual in zip(expectedResults, actualResults):
        if expected == actual:
            score += 1

    return getPercentage(score, len(expectedResults))


expectedResults = np.array([0, 0, 1, 0, 1, 1])
actualResults = np.array([1, 1, 1, 1, 0, 1])

print("{:.2f}".format(getScore(expectedResults, actualResults)) + " %")
