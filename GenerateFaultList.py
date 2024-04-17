import csv
import random
import math
import torch
from functools import reduce

totalInjections = 10000
WEIGHTS_PATH = '/content/drive/MyDrive/Colab Notebooks/vit_iiipet_train_best.pth'

weights = torch.load(WEIGHTS_PATH)['model'].state_dict()


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
            numLayerInjections = math.ceil((result / totalWeights) * totalInjections)
            print(numLayerInjections)

            for currentNumLayerInjection in range(numLayerInjections):
                rowTensorList = []

                for tensorDim in tensorWeightsSizeList:
                    x = random.randint(0, tensorDim - 1)
                    rowTensorList.append(x)

                bit = random.randint(0, 31)
                # CONVERTING THE LIST INTO A STRING
                rowTensorListStr = ' '.join(map(str, rowTensorList))

                # WRITING A NEW ROW IN THE FAULT_LIST CSV FILE
                spamwriter.writerow([globalInjectionCounter] + [layer] + [rowTensorListStr] + [bit])
                globalInjectionCounter += 1
