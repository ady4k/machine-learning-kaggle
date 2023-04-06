# Brain Anomaly Detection - [Kaggle Competition](https://www.kaggle.com/competitions/unibuc-brain-ad) 

## 1. Introduction
---
In this project we have to classify images of cranial CT scans in two different classes: **normal (class 0)** or **anomalied (class 1)**.

We have to resolve this issue using supervised learning, using well-known methods.

Training data consists of 15000 labeled images, validation data cosnsists of 2000 labeled images and the competition test data consists of 5149 examples.

Data consists of 224x224 grayscale images.
<br><br>

## 2. Libraries used
---
- Numpy
- OpenCV (cv2)
- Scikit-Learn with:
  1. MLPClassifier from neural_network
  2. RandoMForestClassifier from ensemble
  3. GridSearchCV from model_selection
  4. classification_report and confusion_matrix from metrics
- ImbalanceLearning with TomekLinks from under_sampling  
<br>

## 3. Data Reading and Preprocessing
---
Data is stored in normal Python arrays which are then converted into 2D arrays for use in Scikit-Learn. The pixels are normalized in the 0,1 interval. 

The training data is resampled with TomekLinks, as it is quite unbalanced.
<br><br>

## 4. Training models
---
The models we train are the Multi-Layer Perceptron classifier and the Random Forest Classifier.

We use GridSearchCV on both models to find the most optimal hyperparameters and use cross validation to get a better result.

The best model is then used to predict the validation data to check its performance. Classification_Report and Confusion_Matrix offer good metrics for unbalanced datasets to assure the score of the model is accurate.
<br><br>

## 5. Evaluating the model
---
The MLP Classifier uses a hard binary 0-1 classification while the RF Classifier uses a probability based classification.

I found 35% probability for the image to pe positive to be the sweet spot in terms of the performance of the model. It's a decent balance of false positives and false negatives.

The final score of the model was around **88%** with the validation test data, getting a **95% f1-score on the majority class (label 0)** and around **60% on the minority class (label 1)**.
<br><br>

## 6. Writing the submission file
---
The test data predictions are written in a text document which are then manually transformed into a csv file. 

The model used for writing the submission files is the RandomForestClassifier as it got the best score.

The probability used for the classification of the test data is **35%**.
<br><br>

## 7. Conclusion and Final Score
---
Both models used and trained by me offer a mediocre performance on the dataset offered by the competition.

The dataset is highly unbalanced with a rating of around **20:1**. I have previously attempted many more models and sampling techniques and weighting, but this seems to offer me the best performance for the simple solution I have to offer.

The public score I got on the Kaggle competition is a mere **0.57558**, atleast placing me in the top 25%.

## **Final score will be posted on 12th of April.**