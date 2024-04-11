# CHECK integerWeights[weightToChange]
import csv
import torch


def model_execution():
    with torch.no_grad():
        loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=32, shuffle=False, num_workers=8,
                                             pin_memory=True)

        total_count = 0
        accurate_count = 0
        # outputTensors = [] -> nel caso volessimo salvare per ogni inferenza l'output delle predictions
        for images, labels in loader:
            images = images.to(device)  # Size (batch_size=32, 3, 224, 224)
            labels = labels.to(device)  # Size (batch_size=32, 1)

            # Execute the prediction for batch_size=32 images
            predictions = model(images)  # Size (batch_size=32,n_classes=37)

            pred_classes = predictions.argmax(axis=1)  # Size (batch_size=32,1)
            # outputTensors.append(pred_classes) -> nel caso volessimo salvare per ogni inferenza l'output delle predictions

            accurate_predictions = pred_classes == labels  # Size (batch_size=32,1) of booleans

            # Accumulate the number of accurate prediction and total predictions
            accurate_count += accurate_predictions.sum().item()
            total_count += len(accurate_predictions)

        # Calculate the accuracy as the proportion of accurate predictions over total predictions
        accuracy = accurate_count / total_count

        return accuracy


def output_file_write(outputDict):
    outputFile = open("InjectionResults.txt", "w")
    for key in outputDict.keys():
        outputFile.write(key + outputDict[key])
    outputFile.close()


def update_weights(floatWeights, weightToChange, bitToChange):
    integerWeights = floatWeights.view(torch.int32)

    bit = 1 << bitToChange  # x << y insert x followed by y times 0
    integerWeights[weightToChange] ^= bit  # sum bit to the previous number
    # to flip back xor again with the same bit

    updatedWeights = integerWeights.view(torch.float32)

    return updatedWeights


# -

WEIGHTS_PATH = '/content/drive/MyDrive/Colab Notebooks/vit_iiipet_train_best.pth'
weights = torch.load(WEIGHTS_PATH)['model'].state_dict()

with open('FaultList.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar=' ')
    outputDict = {}  # Matching every injectionNumber to its evaluation result

    # FOR EACH ROW IN THE CSV FILE
    for row in spamreader:
        # FAULT INJECTION

        injectionNumber = row[0]
        layerInjected = row[1] + ".weight"
        weightToChange = row[2]
        bitToChange = row[3]

        convWeights = weights.get(layerInjected)
        copyWeights = convWeights.clone()  # COPY OF THE ORIGINAL TENSOR ASSOCIATED TO layerInjected

        faultInjectedWeights = update_weights(convWeights, weightToChange, bitToChange)

        with torch.no_grad():
            convWeights.copy_(faultInjectedWeights)
            weights[layerInjected] = convWeights

        model.load_state_dict(weights)

        # INFERENCES EXECUTION
        outputModel = model_execution()

        outputDict[injectionNumber] = outputModel

        # ROLLING BACK TO THE ORIGINAL TENSOR IN ORDER TO AVOID MULTIPLE INJECTIONS -- do xor again
        with torch.no_grad():
            weights[layerInjected] = copyWeights

# WRITING outputDict ON A FILE
output_file_write(outputDict)

# masked tutto uguale
# non_critical top1 uguale, ma il resto del vettore diverso
# critical top 1 cambia
# file con num inienzioni, top1correct, top1 robust, masked, non critical, critical
