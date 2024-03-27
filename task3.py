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
                tensorWeights_size_list=list(tensorWeights.size())

                list=[]

                for t in len(tensorWeights_size_list):
                    x=random.randint(0, tensorWeights_size_list[t]-1)
                    list.append(x)

                spamwriter.writerow([i] + [k] + [list] + [...]) #aggiungere bit da flippare
            i+=1