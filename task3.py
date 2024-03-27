import csv
import random

with open('FaultList.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    i=0

    for k in weights.keys():

        if k.endswith(".weight"):
            tensorWeights = weights.get(k)
            l=random.randint(1,5)
            for j in range(l):
                x=random.randint(0, len(tensorWeights)-1)
                y=random.randint(0, len(tensorWeights[x])-1)
                z=random.randint(0, len(tensorWeights[x][y])-1)
                w=random.randint(0, len(tensorWeights[x][y][z])-1)
            spamwriter.writerow([i] + [k] + ["([x],[y],[z],[w])"], ...)
            i+=1