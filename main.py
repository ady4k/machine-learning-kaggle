import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import TomekLinks

### CITIRE + PREPROCESARE DATE
## Train Data
train_data = open("data/train_labels.txt")  # deschidere fisier cu etichetele imaginilor de antrenare

train_labels = []                         # lista cu etichetele imaginilor de antrenare
train_lines = train_data.readlines()[1:]  # citire linii din fisier, fara prima linie
train_images = []                         # lista cu imagini de antrenare

for line in train_lines:
    current_path = "data/data/" + line[0:6] + ".png"                # citire calea imaginii de pe linia curenta
    train_labels.append(int(line[7]))                               # citire etichete
    current_image = cv2.imread(current_path, cv2.IMREAD_GRAYSCALE)  # citire imagini in grayscale
    current_image = current_image.astype("float32")   / 255.0       # normalizare pixeli de la 0-255 la 0-1
    train_images.append(current_image)                              # adaugare imagine in lista
train_images_number = train_lines.__len__()                         # numarul de imagini de antrenare

## Validation Data
validation_data = open("data/validation_labels.txt")

validation_labels = []
validation_lines = validation_data.readlines()[1:]
validation_images = []

for line in validation_lines:
    current_path = "data/data/" + line[0:6] + ".png"
    validation_labels.append(int(line[7]))
    current_image = cv2.imread(current_path, cv2.IMREAD_GRAYSCALE)
    current_image = current_image.astype("float32") / 255.0
    validation_images.append(current_image)
validation_images_number = validation_lines.__len__()

## Test Data
test_data = open("data/test_sample.txt")

test_lines = test_data.readlines()[1:]
test_images = []

for line in test_lines:
    current_path = "data/data/" + line[0:6] + ".png"
    current_image = cv2.imread(current_path, cv2.IMREAD_GRAYSCALE)
    current_image = current_image.astype("float32") / 255.0
    test_images.append(current_image)
test_images_number = test_lines.__len__()



## Transformare imagini in vectori 2D (15000x50176) pentru scikit-learn
image_height = image_width = 224
train_images_2D = np.reshape(train_images, (train_images_number, image_height * image_width))
validation_images_2D = np.reshape(validation_images, (validation_images_number, image_height * image_width))
test_images_2D = np.reshape(test_images, (test_images_number, image_height * image_width))
print("Pixel Values:")
print("Max: ", np.max(train_images_2D))
print("Min: ", np.min(train_images_2D))
print("Mean: ", np.mean(train_images_2D))
print("Std: ", np.std(train_images_2D))

## Undersampling imagini cu TomekLinks
train_images_resampled, train_labels_resampled = TomekLinks(sampling_strategy="majority").fit_resample(train_images_2D, train_labels)
print(train_images_resampled.shape)



### HIPERPARAMETRII MLP
# hyperparams_mlp = {
#    "hidden_layer_sizes": [(25, 25, 25), (100, 100), (256, 256)],
#    "activation": ["relu", "tanh"],
#    "solver": ["adam", "lbfgs"],
#    "alpha": [0.0001, 0.001]
# }

hyperparams_mlp = {

}

### ANTRENAREA MODELULUI MLPCassifier din cadrul scikit-learn
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=100, alpha=0.0001,
                                   solver="adam", verbose=True, tol=0.0001, random_state=1,
                                   learning_rate_init=0.001)
mlp_classifier_CV = GridSearchCV(mlp_classifier_model, hyperparams_mlp, n_jobs=-1, verbose=4, cv=5)
mlp_classifier_CV.fit(train_images_resampled, train_labels_resampled)

print("Cei mai buni parametrii: ", mlp_classifier_CV.best_params_)
print("Cel mai bun scor: ", mlp_classifier_CV.best_score_)

validation_predictions = mlp_classifier_CV.predict(validation_images_2D)
best_model = mlp_classifier_CV.best_estimator_

print("Rezultat pe validare MLP:")
print(classification_report(validation_labels, validation_predictions))
print(confusion_matrix(validation_labels, validation_predictions))



### ANTRENAREA MODELULUI SVC din cadrul scikit-learn
svc_model = SVC(kernel="rbf", verbose=True, max_iter=-1, C=1, gamma=0.001)
svc_model.fit(train_images_resampled, train_labels_resampled)

validation_predictions = svc_model.predict(validation_images_2D)
print("Rezultat pe validare:")
print(classification_report(validation_labels, validation_predictions))
print(confusion_matrix(validation_labels, validation_predictions))



## HIPERPARAMETRII RFC
#hyperparams_rfc = {
#    "n_estimators": [100, 200, 300],
#    "max_depth": [10, 20, 30],
#    "min_samples_split": [2, 5, 10],
#    "max_features": ["auto", "sqrt", "log2"],
#}

hyperparams_rfc = {

}
## ANTRENAREA MODELULUI RandomForestClassifier din cadrul scikit-learn
rfc_model = RandomForestClassifier(n_estimators=400, verbose=5, n_jobs=-1,
                                  max_features="sqrt", max_depth=20)
rfc_classifier_CV = GridSearchCV(rfc_model, hyperparams_rfc, n_jobs=-1, verbose=4, cv=5)
rfc_classifier_CV.fit(train_images_resampled, train_labels_resampled)

print("Cei mai buni parametrii: ", rfc_classifier_CV.best_params_)
print("Cel mai bun scor: ", rfc_classifier_CV.best_score_)

best_rfc_model = rfc_classifier_CV.best_estimator_
validation_predictions = best_rfc_model.predict_proba(validation_images_2D)

for i in range(2, 6):
    validation_predictions_proba = (validation_predictions[:, 1] >= i / 10).astype(int)
    print("Rezultat pe validare:" + str(i) * 10 + "%:")
    print(classification_report(validation_labels, validation_predictions_proba))
    print(confusion_matrix(validation_labels, validation_predictions_proba))



### PREDICTIA PE SETUL DE TEST SI GENERAREA FISIERULUI DE PREDICTII
## Folosit cu RFC, oferind cel mai bun scor
test_predictions = rfc_model.predict_proba(test_images_2D)
test_predictions_proba = (test_predictions[:, 1] >= 0.35).astype(int)
with open ("predictions.txt", "w") as file:
    file.write("id,class\n")
    for i in range(0, test_images_number):
        file.write(test_lines[i][0:6] + "," + str(test_predictions_proba[i]) + "\n")