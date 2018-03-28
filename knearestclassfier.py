import numpy as np 
from math import sqrt
import warnings

import matplotlib.pyplot as plt 
from matplotlib import style

from collections import Counter
import random
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing, cross_validation, neighbors

def load_input_data():
    df = pd.read_csv("breastCancer.csv")
    df.replace ("?", -9999, inplace = True)
    df.drop (["id"], 1, inplace = True)
    input_data = df.astype(float).values.tolist()
    random.shuffle(input_data)

    return input_data

def KNearestNeighbors(input, reading, kvalue):
    distances = []
    for iterator in input:
        for feature in input[iterator]:
            EuclideanDistance = np.linalg.norm(np.array(feature) - np.array(reading))
            distances.append([EuclideanDistance,iterator])
        votes = [i[1] for i in sorted(distances)[:kvalue]]
        votes_result = Counter(votes).most_common(1)[0][0]
        confidence = Counter(votes).most_common(1)[0][1]/kvalue

    return votes_result, confidence

def test_train_KNN(k):
    
    full_data = load_input_data()
    test_size = 0.2
    train_set = {2: [], 4: []}  # Class: (2 for benign, 4 for malignant)
    test_set = {2: [], 4: []}  # Class: (2 for benign, 4 for malignant)
    train_data = full_data[:-int(test_size * len(full_data))]  # 80% of the data
    test_data = full_data[-int(test_size * len(full_data)):]  # 20% of the data

    for entry in train_data:
        train_set[entry[-1]].append(entry[:-1])  
    
    for entry in test_data:
        test_set[entry[-1]].append(entry[:-1])
    
    correct = 0
    total = 0
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = KNearestNeighbors(train_set, data, k)
            if group == vote:
                correct += 1
            total += 1
    
    Accuracy = correct / total
    
    return Accuracy

k_range = list(range(1,10))
k_scores = []

for k in k_range:
    k_scores.append(test_train_KNN(k))

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Accuracy')
plt.ticklabel_format(style='plain',axis='x',useOffset=False)
plt.show()

def scikitlearn_KNN():
    df = pd.read_csv("breastCancer.csv")
    df.replace ("?", -9999, inplace = True)
    df.drop (["id"], 1, inplace = True)
    input_data = df.astype(float).values.tolist()
    random.shuffle(input_data)
    
    X = np.array(df.drop(['class'],1)) #creates features
    y = np.array(df['class']) #creates labels

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)

    accuracy = clf.score(X_test,y_test)
    print(accuracy)

scikitlearn_KNN()
