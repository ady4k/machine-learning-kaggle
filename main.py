import numpy as np
import cv2

### CITIRE + PREPROCESARE DATE
## Train Data
train = open("data/train_labels.txt")  # deschidere fisier cu etichetele imaginilor de antrenare

trainLabels = []                    # lista cu etichetele imaginilor de antrenare
trainLines = train.readlines()[1:]  # citire linii din fisier, fara prima linie
trainImages = []                    # lista cu imagini de antrenare

for line in trainLines:
    currPath = "data/data/" + line[0:6] + ".png"        # citire calea imaginii de pe linia curenta
    trainLabels.append(int(line[7]))                    # citire etichete
    image = cv2.imread(currPath, cv2.IMREAD_GRAYSCALE)  # citire imagini in grayscale
    image = image.astype('float32') / 255.0             # normalizare pixeli de la 0-255 la 0-1
    trainImages.append(image)                           # adaugare imagine in lista

## Validation Data
validation = open("data/validation_labels.txt")

validationLabels = []
validationLines = validation.readlines()[1:]
validationImages = []

for line in validationLines:
    currPath = "data/data/" + line[0:6] + ".png"
    validationLabels.append(int(line[7]))
    image = cv2.imread(currPath, cv2.IMREAD_GRAYSCALE)
    image = image.astype('float32') / 255.0
    validationImages.append(image)

## Test Data
test = open("data/test_sample.txt")

testLines = test.readlines()[1:]
testImages = []

for line in testLines:
    currPath = "data/data/" + line[0:6] + ".png"
    image = cv2.imread(currPath, cv2.IMREAD_GRAYSCALE)
    image = image.astype('float32') / 255.0
    testImages.append(image)

print(trainImages)