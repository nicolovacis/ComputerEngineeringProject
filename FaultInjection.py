import csv
import argparse
import re
import torch
import torch.nn as nn
import torchvision
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


def generate_output_model(classes_accuracy):
    output_model = {}

    # CALCULATING FOR EACH CLASS THE ACCURACY AND INSERTING IT INTO THE HASHMAP
    for class_name, result in classes_accuracy.items():
        class_accuracy = result[1] / result[0]
        output_model[class_name] = class_accuracy

    return output_model


def model_execution(device, dataset, model, loader):
    with torch.no_grad():
        classes_accuracy = {}

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # EXECUTE THE PREDICTION
            predictions = model(images)

            pred_classes = predictions.argmax(axis=1)

            accurate_predictions = pred_classes == labels

            index = 0

            # CALCULATING TOTAL_IMAGES AND NUMBER OF CORRECT PREDICTIONS FOR EACH CLASS
            for result_prediction in accurate_predictions:

                class_name = dataset.classes[labels[index]]

                if class_name in classes_accuracy:
                    if result_prediction:
                        classes_accuracy[class_name] = [classes_accuracy[class_name][0],
                                                        classes_accuracy[class_name][1] + 1]
                else:
                    if result_prediction:
                        classes_accuracy[class_name] = [1, 1]
                    else:
                        classes_accuracy[class_name] = [1, 0]

                index += 1

        output_model_accuracy = generate_output_model(classes_accuracy)

        return output_model_accuracy


def map_to_array(classes_accuracy):
    classes_accuracy_arr = [(class_name, accuracy) for class_name, accuracy in classes_accuracy.items()]

    classes_sorted_accuracy_arr = sorted(classes_accuracy_arr, key=lambda x: x[1], reverse=True)

    classes_sorted_arr = [item[0] for item in classes_sorted_accuracy_arr]

    return classes_sorted_arr


def calculate_top_one_robust(output_gold_model, output_model):
    min_variation = float("inf")
    top_one_robust_class = None

    for class_name in output_gold_model.keys():
        variation = output_gold_model[class_name] - output_model[class_name]

        if variation < min_variation:
            min_variation = variation
            top_one_robust_class = class_name

    return top_one_robust_class


def calculate_metrics(output_gold_model_arr, output_model):
    output_model_arr = map_to_array(output_model)
    top_one_correct = output_model_arr[0]

    if output_gold_model_arr == output_model_arr:
        return top_one_correct, 1, 0, 0

    if output_gold_model_arr[0] == output_model_arr[0]:
        return top_one_correct, 0, 1, 0

    return top_one_correct, 0, 0, 1


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

    args = parser.parse_args()

    # GETTING THE PARAMETERS FROM CLI
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

    loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=32, shuffle=False, num_workers=8,
                                         pin_memory=True)

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
            output_gold_model = model_execution(device, dataset, model, loader)
            output_gold_model_arr = map_to_array(output_gold_model)

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
                output_model = model_execution(device, dataset, model, loader)

                # METRICS CALCULATION
                top_one_robust = calculate_top_one_robust(output_gold_model, output_model)
                top_one_correct, masked, non_critical, critical = calculate_metrics(output_gold_model_arr, output_model)

                # WRITING THE RESULT FOR THE Nth INJECTION
                spamwriter.writerow([injection_number] + [layer_injected] + [weight_to_change_str] + [bit_to_change]
                                    + [top_one_correct] + [top_one_robust] + [masked]
                                    + [non_critical] + [critical])

                # ROLLING BACK TO THE ORIGINAL TENSOR IN ORDER TO AVOID MULTIPLE INJECTIONS
                original_weights = update_weights(conv_weights, weight_to_change, bit_to_change)
                with torch.no_grad():
                    weights[layer_injected] = original_weights


if __name__ == '__main__':
    main()
