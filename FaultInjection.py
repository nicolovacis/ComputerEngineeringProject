# CHECK integerWeights[weightToChange]
import csv
import torch

loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=32, shuffle=False, num_workers=8,
                                     pin_memory=True)


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


def create_linked_list(classesAccuracy):
    head = None

    # Calculate the accuracy for each class and insert into the linked list
    for className, result in classesAccuracy.items():
        classAccuracy = result[1] / result[0]
        head = insert_node_class(head, className, classAccuracy)

    return head


def insert_node_class(head, nameClass, accuracyClass):
    node = Node((nameClass, accuracyClass))

    if head is None:
        head = node
        return head

    current = head
    prev = None

    while current.next is not None and current.next.data[1] >= accuracyClass:
        prev = current
        current = current.next

    if current.data[1] < accuracyClass:
        if prev is None:
            node.next = head
            head = node
        else:
            prev.next = node
            node.next = current
    else:
        node.next = current.next
        current.next = node

    return head


def model_execution():
    with torch.no_grad():
        total_count = 0
        accurate_count = 0
        classesAccuracy = {
            "cane": [1, 1]
        }
        # outputTensors = [] -> nel caso volessimo salvare per ogni inferenza l'output delle predictions
        for images, labels in loader:
            images = images.to(device)  # Size (batch_size=32, 3, 224, 224)
            labels = labels.to(device)  # Size (batch_size=32, 1)

            # Execute the prediction for batch_size=32 images
            predictions = model(images)  # Size (batch_size=32,n_classes=37)

            pred_classes = predictions.argmax(axis=1)  # Size (batch_size=32,1)
            # outputTensors.append(pred_classes) -> nel caso volessimo salvare per ogni inferenza l'output delle predictions

            accurate_predictions = pred_classes == labels  # Size (batch_size=32,1) of booleans

            # calculating accuracy for each class

            for index, resultPrediction in enumerate(accurate_predictions):

                className = dataset.classes[labels[index]]

                if className in classesAccuracy:
                    if resultPrediction:
                        classesAccuracy[className] = [classesAccuracy[className][0],
                                                      classesAccuracy[className][0][1] + 1]
                else:
                    if resultPrediction:
                        classesAccuracy[className] = [1, 1]
                    else:
                        classesAccuracy[className] = [1, 0]

        head = create_linked_list(classesAccuracy)

        return head


def calculate_output_row(outputGoldModel, outputModel):
    return


def update_weights(floatWeights, weightToChange, bitToChange):
    integerWeights = floatWeights.view(torch.int32)

    bit = 1 << bitToChange  # x << y insert x followed by y times 0
    integerWeights[weightToChange] ^= bit  # sum bit to the previous number
    # to flip back xor again with the same bit

    updatedWeights = integerWeights.view(torch.float32)

    return updatedWeights


WEIGHTS_PATH = 'vit_iiipet_train_best.pth'
weights = torch.load(WEIGHTS_PATH)['model'].state_dict()

with open('FaultListInjection.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

    with open('FaultList.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar=' ')

        # EXECUTING THE GOLD MODEL
        model.load_state_dict(weights)
        outputGoldModel = model_execution()

        # FOR EACH INJECTION
        for injection in spamreader:
            injectionNumber = injection[0]
            layerInjected = injection[1] + ".weight"
            weightToChange = injection[2]
            bitToChange = injection[3]
            convWeights = weights.get(layerInjected)
            copyWeights = convWeights.clone()  # COPY OF THE ORIGINAL TENSOR ASSOCIATED TO layerInjected

            faultInjectedWeights = update_weights(convWeights, weightToChange, bitToChange)

            with torch.no_grad():
                convWeights.copy_(faultInjectedWeights)
                weights[layerInjected] = convWeights

            model.load_state_dict(weights)

            # INFERENCES EXECUTION
            outputModel = model_execution()

            injectionOutputResults = calculate_output_row(outputGoldModel, outputModel)

            spamwriter.writerow([injectionNumber] + [layerInjected] + [weightToChange] + [bitToChange]
                                + [injectionOutputResults[0]] + [injectionOutputResults[1]] + [injectionOutputResults[2]]
                                + [injectionOutputResults[3]] + [injectionOutputResults[4]])

            # ROLLING BACK TO THE ORIGINAL TENSOR IN ORDER TO AVOID MULTIPLE INJECTIONS -- do xor again
            with torch.no_grad():
                weights[layerInjected] = copyWeights


# masked tutto uguale
# non_critical top1 uguale, ma il resto del vettore diverso
# critical top 1 cambia
# file con num inienzioni, top1correct, top1 robust, masked, non critical, critical
