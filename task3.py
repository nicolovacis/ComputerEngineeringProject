import csv
import random

with open('FaultList.csv', 'w', newline='') as csvfile:
    # CREAZIONE FILE CSV
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    i=0

    for k in weights.keys():

        # ITERO SU TUTTE LE CHIAVI/LAYER CHE TERMINANO CON WEIGHT - sono dei pesi
        if k.endswith(".weight"):

            # PER OGNI LAYER OTTENGO I TENSORI ASSOCIATI
            tensorWeights = weights.get(k)

            # ITERO SU OGNI LAYER IN MANIERA RANDOMICA TRA a E b
            l=random.randint(1,5)
            for j in range(l):

                # OTTENGO IL TENSORE IN FORMATO LISTA FORMATA DA INT --> [4,2,3,5]
                tensorWeights_size_list=list(tensorWeights.size())

                # OGNI TENSORE PUO' AVERE DIM DIVERSA --> CALCOLO IN MANIERA DINAMICA/RANDOMICA L'ELEMENTO SINGOLO DEL TENSORE
                list=[]

                for t in len(tensorWeights_size_list):
                    x=random.randint(0, tensorWeights_size_list[t]-1)
                    list.append(x)

                # AGGIUNGO LA RIGA AL CSV
                spamwriter.writerow([i] + [k] + [list] + [...]) #aggiungere bit da flippare

            i+=1