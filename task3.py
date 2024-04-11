import csv
import random

with open('FaultList.csv', 'w', newline='') as csvfile:
    # CREAZIONE FILE CSV
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)

    i=0

    for k in range(10):#ipotizzo 10 layer coi pesi
        l = random.randint(1, 5)
        for j in range(l):
            # OTTENGO IL TENSORE IN FORMATO LISTA FORMATA DA INT --> [4,2,3,5]
            #tensorWeights_size_list = list(tensorWeights.size())
            tensorWeights_size_list = [768, 10, 97]
            # OGNI TENSORE PUO' AVERE DIM DIVERSA --> CALCOLO IN MANIERA DINAMICA/RANDOMICA L'ELEMENTO SINGOLO DEL TENSORE
            listOut = []

            for dimTensor in tensorWeights_size_list:
                x = random.randint(0, dimTensor - 1)
                listOut.append(x)

            bit = random.randint(0, 31)

            # AGGIUNGO LA RIGA AL CSV
            spamwriter.writerow([i] + [k] + [listOut] + [bit])  # aggiungere bit da flippare
            i += 1

# media pesata al posto di random tra a e b -> il totale è gli elementi della lista moltiplicati 4x3x2x5