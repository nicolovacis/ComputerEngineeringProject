import csv
import argparse
import math
import os
import re
import torch
import torch.nn as nn
import torchvision
import numpy
from tqdm import tqdm


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()
        # Wrapped dataset
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def model_execution(device, model, loader):
    with torch.no_grad():
        counter_top_one_correct = 0
        outputPredictions = []

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # EXECUTE THE PREDICTION
            predictions = model(images)

            outputPredictions.append(predictions)

            pred_classes = predictions.argmax(axis=1)

            accurate_predictions = pred_classes == labels

            counter_top_one_correct += accurate_predictions.sum().item()

        return outputPredictions, counter_top_one_correct


def read_numbers(path_txt):
    numbers = []

    # OPENING FILE
    with open(path_txt, 'r') as txtfile:
        # Reading each line from the txt file
        for row in txtfile:
            numbers.append(int(row.strip()))

    return numbers


def write_numbers(txt_path, dataset_len, desired_size):
    selected_dataset_images = numpy.random.choice(dataset_len, size=desired_size, replace=False)

    with open(txt_path, 'w') as file:
        for image in selected_dataset_images:
            file.write(str(image) + '\n')


def calculate_metrics(output_gold_model, output_model):
    counter_masked = 0
    counter_non_critical = 0
    counter_critical = 0

    for i in range(len(output_gold_model)):
        gold_model_tensor = output_gold_model[i]
        injected_model_tensor = output_model[i]

        for j in range(gold_model_tensor.shape[0]):
            classes_probabilty_output_gold = gold_model_tensor[j, :]
            classes_probabilty_output = injected_model_tensor[j, :]

            # Perform element-wise comparison
            is_masked = numpy.array_equal(classes_probabilty_output_gold, classes_probabilty_output)

            if is_masked:
                counter_masked += 1
            else:
                is_non_critical = classes_probabilty_output_gold.argmax() == classes_probabilty_output.argmax()

                if is_non_critical:
                    counter_non_critical += 1
                else:
                    counter_critical += 1

    return counter_masked, counter_non_critical, counter_critical


def update_weights(float_weights, weight_to_change, bit_to_change):
    integer_weights = float_weights.view(torch.int32)

    indices = tuple(weight_to_change)

    # x << y INSERT x FOLLOWERD BY y TIMES 0
    bit = 1 << bit_to_change
    integer_weights[indices] ^= bit

    updated_weights = integer_weights.view(torch.float32)

    return updated_weights


def main():
    parser = argparse.ArgumentParser(
        prog='GenerateFaultList',
        description='Script that generates a fault list',
    )

    parser.add_argument('--dataset_percentage', type=int, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--csv_input_path', type=str, required=True)
    parser.add_argument('--csv_output_path', type=str, required=True)
    parser.add_argument('--numbers_selected_txt_path', type=str, required=True)

    args = parser.parse_args()

    # GETTING THE PARAMETERS FROM CLI
    numbers_selected_txt_path = args.numbers_selected_txt_path
    dataset_percentage = args.dataset_percentage
    DATASET_PATH = args.dataset_path
    WEIGHTS_PATH = args.weights_path
    csv_input_path = args.csv_input_path
    csv_output_path = args.csv_output_path

    # INITIALIZATION
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = torchvision.datasets.OxfordIIITPet(DATASET_PATH, split='test', download=True)
    n_classes = len(dataset.classes)

    model = torchvision.models.vit_b_16()

    model.heads.head = nn.Linear(in_features=768, out_features=n_classes, bias=True)

    transform = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()

    transformed_dataset = TransformedDataset(dataset, transform=transform)

    dataset_len = len(transformed_dataset)

    desired_size = math.ceil((dataset_percentage / 100) * dataset_len)

    # WRITING A NEW FILE ONLY IF IT DOESN'T ALREADY EXIST
    if not os.path.exists(numbers_selected_txt_path):
        write_numbers(numbers_selected_txt_path, dataset_len, desired_size)

    list_selected_numbers = read_numbers(numbers_selected_txt_path)

    sample_size = len(list_selected_numbers)

    loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=32, shuffle=False, num_workers=8,
                                         pin_memory=True, sampler=list_selected_numbers)

    # SETTING THE MODEL TO EVALUATION MODE
    model.eval()
    model.to(device)

    # LOADING THE WEIGHTS
    weights = torch.load(WEIGHTS_PATH)['model'].state_dict()

    with open(csv_output_path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

        with open(csv_input_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar=' ')

            total_rows = sum(1 for row in spamreader)
            csvfile.seek(0)
            loader_progress_bar = tqdm(spamreader, total=total_rows, desc='Evaluating', colour='green')

            # EXECUTING THE GOLD MODEL
            model.load_state_dict(weights)
            output_gold_model, _ = model_execution(device, model, loader)

            # WRITING THE RESULT FOR THE Nth INJECTION
            header = ['inj_id', 'layer_injected', 'weight_coords', 'bit_to_change', 'desired_size', 'top_one_correct', 'masked', 'non_critical', 'critical']
            spamwriter.writerow(header)

            # FOR EACH INJECTION IN THE FAULT LIST
            for injection in loader_progress_bar:
                injection_number = injection[0]
                layer_injected = injection[1]
                weight_to_change_str = injection[2]
                weight_to_change = list(map(int, re.split(r'\s+', weight_to_change_str.strip())))
                bit_to_change = int(injection[3])

                conv_weights = weights.get(layer_injected)

                fault_injected_weights = update_weights(conv_weights, weight_to_change, bit_to_change)
                with torch.no_grad():
                    conv_weights.copy_(fault_injected_weights)
                    weights[layer_injected] = conv_weights

                model.load_state_dict(weights)

                # INJECTED MODEL EXECUTION
                output_model, top_one_correct = model_execution(device, model, loader)

                # METRICS CALCULATION
                masked, non_critical, critical = calculate_metrics(output_gold_model, output_model)

                # WRITING THE RESULT FOR THE Nth INJECTION
                spamwriter.writerow([injection_number] + [layer_injected] + [weight_to_change_str] + [bit_to_change] +
                                    [sample_size] + [top_one_correct] + [masked] + [non_critical] + [critical])

                # ROLLING BACK TO THE ORIGINAL TENSOR IN ORDER TO AVOID MULTIPLE INJECTIONS
                original_weights = update_weights(conv_weights, weight_to_change, bit_to_change)
                with torch.no_grad():
                    weights[layer_injected] = original_weights


if __name__ == '__main__':
    main()
