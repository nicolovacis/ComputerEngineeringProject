import csv
import re
import torch
import torch.nn as nn
import torchvision


# START SETTING
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


DATASET_PATH = 'downloaded_datasets/oxford_iiit_pet'

dataset = torchvision.datasets.OxfordIIITPet(DATASET_PATH, split='test', download=True)

n_classes = len(dataset.classes)

model = torchvision.models.vit_b_16()

image_width = 224
image_height = 224

model_input_size = (3, image_width, image_height)
model.heads.head = nn.Linear(in_features=768, out_features=n_classes, bias=True)

transform = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()

transformed_dataset = TransformedDataset(dataset, transform=transform)

loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

model.eval()

model.to(device)

totalInjections = 10000

WEIGHTS_PATH = '/content/drive/MyDrive/Colab Notebooks/vit_iiipet_train_best.pth'

weights = torch.load(WEIGHTS_PATH)['model'].state_dict()


# END SETTING
def generate_output_model(classesAccuracy):
    outputModel = {}

    # CALCULATING FOR EACH CLASS THE ACCURACY AND INSERTING IT INTO THE HASHMAP
    for className, result in classesAccuracy.items():
        classAccuracy = result[1] / result[0]
        outputModel[className] = classAccuracy

    return outputModel


def model_execution():
    with torch.no_grad():
        classesAccuracy = {}

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # EXECUTE THE PREDICTION
            predictions = model(images)

            pred_classes = predictions.argmax(axis=1)

            accurate_predictions = pred_classes == labels

            # CALCULATING TOTAL_IMAGES AND NUMBER OF CORRECT PREDICTIONS FOR EACH CLASS
            for index, resultPrediction in enumerate(accurate_predictions):

                className = dataset.classes[labels[index]]

                if className in classesAccuracy:
                    if resultPrediction:
                        classesAccuracy[className] = [classesAccuracy[className][0], classesAccuracy[className][1] + 1]
                else:
                    if resultPrediction:
                        classesAccuracy[className] = [1, 1]
                    else:
                        classesAccuracy[className] = [1, 0]

        outputModelAccuracy = generate_output_model(classesAccuracy)

        return outputModelAccuracy


def map_to_array(classesAccuracy):
    classesAccuracyArr = [(className, accuracy) for className, accuracy in classesAccuracy.items()]

    classesSortedAccuracyArr = sorted(classesAccuracyArr, key=lambda x: x[1], reverse=True)

    classesSortedArr = [item[0] for item in classesSortedAccuracyArr]

    return classesSortedArr


def calculate_top_one_robust(outputGoldModel, outputModel):
    minVariation = float("inf")
    topOneRobustClass = None

    for className in outputGoldModel.keys():
        variation = outputGoldModel[className] - outputModel[className]

        if variation < minVariation:
            minVariation = variation
            topOneRobustClass = className

    return topOneRobustClass


def calculate_metrics(outputGoldModelArr, outputModel):
    outputModelArr = map_to_array(outputModel)
    topOneCorrect = outputModelArr[0]

    if outputGoldModelArr == outputModelArr:
        return topOneCorrect, 1, 0, 0

    if outputGoldModelArr[0] == outputModelArr[0]:
        return topOneCorrect, 0, 1, 0

    return topOneCorrect, 0, 0, 1


def update_weights(floatWeights, weightToChange, bitToChange):
    integerWeights = floatWeights.view(torch.int32)

    indices = tuple(weightToChange)

    # x << y INSERT x FOLLOWERD BY y TIMES 0
    bit = 1 << bitToChange
    integerWeights[indices] ^= bit

    updatedWeights = integerWeights.view(torch.float32)

    return updatedWeights


with open('FaultListInjection.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

    with open('/content/drive/MyDrive/Colab Notebooks/fl.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar=' ')

        # EXECUTING THE GOLD MODEL
        model.load_state_dict(weights)
        outputGoldModel = model_execution()
        outputGoldModelArr = map_to_array(outputGoldModel)

        # FOR EACH INJECTION
        for injection in spamreader:
            injectionNumber = injection[0]
            layerInjected = injection[1]
            weightToChangeStr = injection[2]
            weightToChange = list(map(int, re.split(r'\s+', weightToChangeStr.strip())))
            bitToChange = int(injection[3])
            convWeights = weights.get(layerInjected)

            faultInjectedWeights = update_weights(convWeights, weightToChange, bitToChange)

            with torch.no_grad():
                convWeights.copy_(faultInjectedWeights)
                weights[layerInjected] = convWeights

            model.load_state_dict(weights)

            # INJECTED MODEL EXECUTION
            outputModel = model_execution()

            # METRICS CALCULATION
            topOneRobust = calculate_top_one_robust(outputGoldModel, outputModel)

            topOneCorrect, masked, nonCritical, critical = calculate_metrics(outputGoldModelArr, outputModel)

            # WRITING THE RESULT FOR THE Nth INJECTION
            spamwriter.writerow([injectionNumber] + [layerInjected] + [weightToChangeStr] + [bitToChange]
                                + [topOneCorrect] + [topOneRobust] + [masked]
                                + [nonCritical] + [critical])

            # ROLLING BACK TO THE ORIGINAL TENSOR IN ORDER TO AVOID MULTIPLE INJECTIONS
            originalWeights = update_weights(convWeights, weightToChange, bitToChange)
            with torch.no_grad():
                weights[layerInjected] = originalWeights
