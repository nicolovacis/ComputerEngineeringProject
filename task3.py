import csv
import random

with open('FaultList.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    i=0

    for k in weights.keys():

        if k.endswith(".weight"):
            tensorWeights = weights.get(k)
            l=random.randint(1,20)
            for j in range(l):
                tensorWeights_size_list=list(tensorWeights.size())

                if(len(tensorWeights_size_list)==4):
                    x=random.randint(0, tensorWeights_size_list[0]-1)
                    y=random.randint(0, tensorWeights_size_list[1]-1)
                    z=random.randint(0, tensorWeights_size_list[2]-1)
                    w=random.randint(0, tensorWeights_size_list[3]-1)
                    spamwriter.writerow([i] + [k] + ["([x],[y],[z],[w])"], ...)
            i+=1