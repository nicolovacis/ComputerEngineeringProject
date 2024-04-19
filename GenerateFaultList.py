import csv
import random
import math
import torch
from functools import reduce




def calc_total_weights(dict_weights):
    weights_number = 0

    for layer in dict_weights.keys():

        if layer.endswith(".weight"):
            tensor_weights = weights.get(layer)
            tensor_weights_size_list = list(tensor_weights.size())

            # MULTIPLYING THE ELEMENTS OF THE SIZE LIST
            result = reduce(lambda x, y: x * y, tensor_weights_size_list)

            weights_number += result

    return weights_number



def main():
    # Paramterizzate

    # Nome file output
    # Total injections

    total_injections = 10000
    WEIGHTS_PATH = 'weights/vit_iiipet_train_best.pth'

    weights = torch.load(WEIGHTS_PATH)['model'].state_dict()
    total_weights = calc_total_weights(weights)

    with open('FaultList.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=' ', quoting=csv.QUOTE_MINIMAL)

        global_injection_counter = 0

        for layer in weights.keys():

            if layer.endswith(".weight"):
                tensor_weights = weights.get(layer)
                tensor_weights_size_list = list(tensor_weights.size())

                # MULTIPLYING THE ELEMENTS OF THE SIZE LIST
                result = reduce(lambda x, y: x * y, tensor_weights_size_list)

                # GETTING HOW MANY INJECTIONS SHOULD I DO ON THE CURRENT LAYER
                num_layer_injections = math.ceil((result / total_weights) * total_injections)

                for current_num_layer_injection in range(num_layer_injections):
                    row_tensor_list = []

                    for tensorDim in tensor_weights_size_list:
                        x = random.randint(0, tensorDim - 1)
                        row_tensor_list.append(x)

                    bit = random.randint(0, 31)
                    # CONVERTING THE LIST INTO A STRING
                    row_tensor_list_str = ' '.join(map(str, row_tensor_list))

                    # WRITING A NEW ROW IN THE FAULT_LIST CSV FILE
                    spamwriter.writerow([global_injection_counter] + [layer] + [row_tensor_list_str] + [bit])
                    global_injection_counter += 1



if __name__ == '__main__':
    main()