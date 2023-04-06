import numpy as np
import cv2

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import TomekLinks

##### Citire si preprocesare date #####
# 1. Datele de antrenare
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

# 2. Datele de validare
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

# 3. Imaginile pentru testare
test_data = open("data/test_sample.txt")

test_lines = test_data.readlines()[1:]
test_images = []

for line in test_lines:
    current_path = "data/data/" + line[0:6] + ".png"
    current_image = cv2.imread(current_path, cv2.IMREAD_GRAYSCALE)
    current_image = current_image.astype("float32") / 255.0
    test_images.append(current_image)
test_images_number = test_lines.__len__()


# 4. Transformare imagini in vectori 2D (15000x50176) pentru scikit-learn
image_height = image_width = 224
train_images_2D = np.reshape(train_images, (train_images_number, image_height * image_width))
validation_images_2D = np.reshape(validation_images, (validation_images_number, image_height * image_width))
test_images_2D = np.reshape(test_images, (test_images_number, image_height * image_width))

print("Valorile Pixelilor:")
print("Max: ", np.max(train_images_2D))
print("Min: ", np.min(train_images_2D))
print("Mean: ", np.mean(train_images_2D))
print("Std: ", np.std(train_images_2D))
print ("Numarul imaginilor de antrenare inainte de sampling: ", len(train_labels))

# 5. Undersampling imagini cu TomekLinks
train_images_resampled, train_labels_resampled = TomekLinks(sampling_strategy="majority").fit_resample(train_images_2D, train_labels)
print ("Numarul imaginilor de antrenare dupa TomekLinks: ", len(train_labels_resampled))


##### MLPClassifier #####
# 1. Hiperparametrii
hyperparams_mlp = {
    "hidden_layer_sizes": [(25, 25, 25), (100, 100), (256, 256)],
    "activation": ["relu", "tanh"],
    "solver": ["adam", "lbfgs"],
    "alpha": [0.0001, 0.001]
}

# 2. Antrenarea modelului
mlp_classifier_model = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=100, alpha=0.0001,
                                   solver="adam", verbose=True, tol=0.0001, random_state=1,
                                   learning_rate_init=0.001)

mlp_classifier_CV = GridSearchCV(mlp_classifier_model, hyperparams_mlp, n_jobs=-1, verbose=4, cv=5)
mlp_classifier_CV.fit(train_images_resampled, train_labels_resampled)

# 3. Evaluarea modelului
print("Cei mai buni parametrii: ", mlp_classifier_CV.best_params_)
print("Cel mai bun scor: ", mlp_classifier_CV.best_score_)

validation_predictions = mlp_classifier_CV.predict(validation_images_2D)
best_model = mlp_classifier_CV.best_estimator_

# 4. Raportul de evaluare si matricea de confuzie
print("Rezultat pe validare MLP:")
print(classification_report(validation_labels, validation_predictions))
print("Matricea de confuzie:")
print(confusion_matrix(validation_labels, validation_predictions))


##### RandomForestClassifier #####
# 1. Hiperparametrii 
hyperparams_rfc = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "max_features": ["auto", "sqrt", "log2"],
}

# 2. Antrenarea modelului
rfc_model = RandomForestClassifier(n_estimators=200, verbose=5, n_jobs=-1,
                                  max_features="sqrt", max_depth=20)

rfc_classifier_CV = GridSearchCV(rfc_model, hyperparams_rfc, n_jobs=-1, verbose=4, cv=5)
rfc_classifier_CV.fit(train_images_resampled, train_labels_resampled)

# 3. Evaluarea modelului
print("Cei mai buni parametrii: ", rfc_classifier_CV.best_params_)
print("Cel mai bun scor: ", rfc_classifier_CV.best_score_)

best_rfc_model = rfc_classifier_CV.best_estimator_
validation_predictions = best_rfc_model.predict_proba(validation_images_2D)

# 4. Raportul de evaluare si matricea de confuzie bazate pe probabilitatea de a fi pozitiv
for i in range(1, 9):
    validation_predictions_proba = (validation_predictions[:, 1] >= i / 10).astype(int)
    print("Rezultat pe validare RFC cu probabilitatea " + str(i * 10) + "%:")
    print(classification_report(validation_labels, validation_predictions_proba))
    print("Matricea de confuzie:")
    print(confusion_matrix(validation_labels, validation_predictions_proba))



##### Predictii pe test + scriere in fisier #####
# Folosit cu RFC, oferind cel mai bun scor
test_predictions = best_rfc_model.predict_proba(test_images_2D)
test_predictions_proba = (test_predictions[:, 1] >= 0.35).astype(int)
with open ("documents/predictions.txt", "w") as file:
    file.write("id,class\n")
    for i in range(0, test_images_number):
        file.write(test_lines[i][0:6] + "," + str(test_predictions_proba[i]) + "\n")