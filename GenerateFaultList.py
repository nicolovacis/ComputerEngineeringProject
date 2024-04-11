import csv
import random
from functools import reduce

totalInjections = 10000
weights = {}


def calc_total_weights(dictWeights):
    weightsNumber = 0

    for layer in dictWeights.keys():

        if layer.endswith(".weight"):
            tensorWeights = weights.get(layer)
            tensorWeightsSizeList = list(tensorWeights.size())

            # MULTIPLYING THE ELEMENTS OF THE SIZE LIST
            result = reduce(lambda x, y: x * y, tensorWeightsSizeList)

            weightsNumber += result

    return weightsNumber


totalWeights = calc_total_weights(weights)

with open('FaultList.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)

    globalInjectionCounter = 0

    for layer in weights.keys():

        if layer.endswith(".weight"):
            tensorWeights = weights.get(layer)
            tensorWeightsSizeList = list(tensorWeights.size())

            # MULTIPLYING THE ELEMENTS OF THE SIZE LIST
            result = reduce(lambda x, y: x * y, tensorWeightsSizeList)

            # GETTING HOW MANY INJECTIONS SHOULD I DO ON THE CURRENT LAYER
            numLayerInjections = (result / totalWeights) * totalInjections
            rowTensorList = []

            for currentNumLayerInjection in range(numLayerInjections):

                for tensorDim in tensorWeightsSizeList:
                    x = random.randint(0, tensorDim - 1)
                    rowTensorList.append(x)

                    bit = random.randint(0, 31)

                    # WRITING A NEW ROW IN THE FAULT_LIST CSV FILE
                    spamwriter.writerow([globalInjectionCounter] + [layer] + [rowTensorList] + [bit])
                    globalInjectionCounter += 1
