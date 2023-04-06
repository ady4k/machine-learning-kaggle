import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

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
    image = image.astype('float32')   / 255.0          # normalizare pixeli de la 0-255 la 0-1
    trainImages.append(image)                           # adaugare imagine in lista
trainImagesNumber = trainLines.__len__()                 # numarul de imagini de antrenare

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
valImagesNumber = validationLines.__len__()

## Test Data
test = open("data/test_sample.txt")

testLines = test.readlines()[1:]
testImages = []

for line in testLines:
    currPath = "data/data/" + line[0:6] + ".png"
    image = cv2.imread(currPath, cv2.IMREAD_GRAYSCALE)
    image = image.astype('float32') / 255.0
    testImages.append(image)
testImagesNumber = testLines.__len__()

## Transformare imagini in vectori 2D (15000x50176) pentru scikit-learn
imageHeight = imageWidth = 224
trainImages2D = np.reshape(trainImages, (trainImagesNumber, imageHeight * imageWidth))
validationImages2D = np.reshape(validationImages, (valImagesNumber, imageHeight * imageWidth))
testImages2D = np.reshape(testImages, (testImagesNumber, imageHeight * imageWidth))
print("max: ", np.max(trainImages2D))
print("min: ", np.min(trainImages2D))
print("mean: ", np.mean(trainImages2D))
print("std: ", np.std(trainImages2D))

  
train_images_resampled, train_labels_resampled = TomekLinks(sampling_strategy='majority').fit_resample(trainImages2D, trainLabels)
print(train_images_resampled.shape)



### PROCESARE HIPERPARAMETRII
# hyperparams = {
#    'hidden_layer_sizes': [(25, 25, 25), (100, 100), (256, 256)],
#    'activation': ['relu', 'tanh'],
#    'solver': ['adam', 'lbfgs'],
#    'alpha': [0.0001, 0.001]
# }

hyperparams = {

}

### ANTRENAREA MODELULUI MLPCassifier din cadrul scikit-learn
mlpClassifierModel = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=100, alpha=1e-4,
                                   solver='adam', verbose=10, tol=1e-4, random_state=1,
                                   learning_rate_init=0.001)

classifier = GridSearchCV(mlpClassifierModel, hyperparams, n_jobs=-1, verbose=10)
classifier.fit(train_images_resampled, train_labels_resampled)

print("Cei mai buni parametrii: ", classifier.best_params_)
print("Cel mai bun scor: ", classifier.best_score_)

#validationPredictions = classifier.predict(validationImages2D)
bestModel = classifier.best_estimator_
validationPredictions = bestModel.predict_proba(validationImages2D)
#print('Rezultat pe validare:')
#print(classification_report(validationLabels, validationPredictions))

### ANTRENAREA MODELULUI SVC din cadrul scikit-learn
#svcModel = SVC(kernel='rbf', verbose=3, max_iter=-1, C=1, gamma=1e-3)
#svcModel.fit(train_images_resampled, train_labels_resampled)

#validationPredictions = svcModel.predict(validationImages2D)
#print('Rezultat pe validare:')
#print(classification_report(validationLabels, validationPredictions))

## Hyperparameters pentru RandomForestClassifier
#hyperparams = {
#    'n_estimators': [100, 200, 300],
#    'max_depth': [10, 20, 30],
#    'min_samples_split': [2, 5, 10],
#    'max_features': ['auto', 'sqrt', 'log2'],
#}

#hyperparams = {
#
#}
## ANTRENAREA MODELULUI RandomForestClassifier din cadrul scikit-learn
#rfcModel = RandomForestClassifier(n_estimators=400, verbose=10, n_jobs=-1,
#                                  max_features=150, max_depth=20)
#rfcModel.fit(train_images_resampled, train_labels_resampled)

#classifier = GridSearchCV(rfcModel, hyperparams, n_jobs=-1, verbose=10, cv=5)
#classifier.fit(train_images_resampled, train_labels_resampled)

#print("Cei mai buni parametrii: ", classifier.best_params_)
#print("Cel mai bun scor: ", classifier.best_score_)

#validationPredictions = rfcModel.predict_proba(validationImages2D)
#bestModel = classifier.best_estimator_
#validationPredictions = bestModel.predict_proba(validationImages2D)
for i in range(2, 6):
    valPred = (validationPredictions[:, 1] >= i / 10).astype(int)
    print('Rezultat pe validare:' + str(i))
    print(classification_report(validationLabels, valPred))
    print(confusion_matrix(validationLabels, valPred))
#valPred = (validationPredictions[:, 1] >= 0.4).astype(int)

#print('Rezultat pe validare:')
#print(classification_report(validationLabels, valPred))
#print(confusion_matrix(validationLabels, valPred))

#testPredictions = rfcModel.predict_proba(testImages2D)
#testPred = (testPredictions[:, 1] >= 0.35).astype(int)
#with open ('predictions.txt', 'w') as f:
#    f.write('id,class\n')
#    for i in range(0, testImagesNumber):
#        f.write(testLines[i][0:6] + ',' + str(testPred[i]) + '\n')


