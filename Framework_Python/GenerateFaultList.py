import csv
import argparse
import random
import math
import torch
from functools import reduce


def calc_total_weights(dict_weights):
    weights_number = 0

    for layer in dict_weights.keys():

        if layer.endswith(".weight"):
            tensor_weights = dict_weights.get(layer)
            tensor_weights_size_list = list(tensor_weights.size())

            # MULTIPLYING THE ELEMENTS OF THE SIZE LIST
            result = reduce(lambda x, y: x * y, tensor_weights_size_list)

            weights_number += result

    return weights_number


def main():
    parser = argparse.ArgumentParser(
        prog='GenerateFaultList',
        description='Script that generates a fault list',
    )

    parser.add_argument('--injections_number', type=int, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--csv_output_path', type=str, required=True)

    args = parser.parse_args()

    WEIGHTS_PATH = args.weights_path
    injections_number = args.injections_number
    csv_output_path = args.csv_output_path

    weights = torch.load(WEIGHTS_PATH)['model'].state_dict()

    total_weights = calc_total_weights(weights)

    with open(csv_output_path, 'w', newline='') as csvfile:
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
                num_layer_injections = math.ceil((result / total_weights) * injections_number)

                for current_num_layer_injection in range(num_layer_injections):
                    row_tensor_list = []

                    for tensor_dim in tensor_weights_size_list:
                        x = random.randint(0, tensor_dim - 1)
                        row_tensor_list.append(x)

                    bit = random.randint(0, 31)
                    # CONVERTING THE LIST INTO A STRING
                    row_tensor_list_str = ' '.join(map(str, row_tensor_list))

                    # WRITING A NEW ROW IN THE FAULT_LIST CSV FILE
                    spamwriter.writerow([global_injection_counter] + [layer] + [row_tensor_list_str] + [bit])
                    global_injection_counter += 1


if __name__ == '__main__':
    main()
