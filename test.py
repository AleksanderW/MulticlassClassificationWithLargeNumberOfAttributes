import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(os.path.join("dataset/csv/ALL_GSE13425_FINAL.csv"), delimiter=";"
)
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1].values
print(X)
print(set(y), len(set(y)))
print(X.shape)

enc = LabelEncoder()
enc.fit(y)
print(y)
y= enc.transform(y)
print(y)
print(set(y), len(set(y)))
# print(data)
#
# # plt.plot([1,2,3])
# # plt.show()
#



X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=13,
                                                    shuffle=True,
                                                    train_size=0.5, stratify=y)


plt.hist(y_train, 7, histtype='bar',  stacked=True, fill=True)
plt.hist(y_test, 7, histtype='bar',  stacked=True, fill=True)
plt.yticks(range(0,50,5))
plt.show()
# if set(y_train).difference(set(y_test)) is not {}:
#     raise Exception("Zbiór testowy nie zawiera klasy ze zbioru treningowego")


