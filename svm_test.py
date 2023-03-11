# Basic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn modules & classes
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics

# Load the data set; In this example, the breast cancer dataset is loaded.
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

# Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Instantiate the Support Vector Classifier (SVC)
svc = SVC(C=1.0, random_state=1, kernel='linear')

# Fit the model
svc.fit(X_train_std, y_train)

# Make the predictions
y_predict = svc.predict(X_test_std)

# Measure the performance
print("Accuracy score %.3f" % metrics.accuracy_score(y_test, y_predict))
