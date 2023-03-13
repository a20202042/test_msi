import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, \
    precision_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('data.csv')
df.columns = [i for i in range(df.shape[1])]
print(df)
df = df.rename(columns={63: 'Output'})
print(df)
X = df.iloc[:, :-1]
print("Features shape =", X.shape)

Y = df.iloc[:, -1]
print("Labels shape =", Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
svm = SVC(C=10, gamma=10, kernel='rbf', max_iter=10000)
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

print(y_pred)
print(svm.score(x_train, y_train))
print(svm.score(x_test, y_test))

# ------繪圖
cf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
precision = precision_score(y_test, y_pred, average='micro')
print(f1, recall, precision)

labels = sorted(list(set(df['Output'])))
labels = [str(labels[0]), str(labels[1])]
labels = [x.upper() for x in labels]

fig, ax = plt.subplots(figsize=(12, 12))

ax.set_title("w")

maping = sns.heatmap(cf_matrix,
                     annot=True,
                     cmap=plt.cm.Blues,
                     linewidths=.2,
                     xticklabels=labels,
                     yticklabels=labels, vmax=8,
                     fmt='g',
                     ax=ax
                     )

plt.savefig('savefig.png')

# 儲存模型

from joblib import dump, load

dump(svm, 'svm_hand.joblib')
svm = load('svm_hand.joblib')
y_pred = svm.predict(x_test)
print(y_pred)
