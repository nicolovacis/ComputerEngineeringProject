import csv


with open('FaultList.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range (5):
        spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
