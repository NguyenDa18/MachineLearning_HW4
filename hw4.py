import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA

# Read in data
lines_data = []
with open('histograms.txt', 'rt') as in_file:
    for line in in_file:
        lines_data.append(line.rstrip('\n'))

# parse output +-1:[data] into x = [1,1,-1,...] y = [data]
data_list = []
for line in lines_data:
    label = int(re.split(':', line)[0])

    parsed_input = re.split('1:', line)[1].strip()
    parsed_input = parsed_input.replace("[", "").replace("]", "")
    parsed_input_list = parsed_input.split(',')
    inputs = [float(el) for el in parsed_input_list]
    row = [label] + inputs
    data_list.append(row)
data = np.asarray(data_list)

x = data[:, 1:]
y = data[:, 0]  #labels
y = y.reshape(5699, )

sample_x = data[0:100, 1:]
sample_y = y[0:100]

# Build the SVM
clf = svm.SVC(kernel='linear')
clf.fit(x, y)
y_pred = clf.predict(sample_x)

scores = zip(sample_y, y_pred)
print(scores)

# Model Evaluation
print(accuracy_score(sample_y, y_pred))
print(classification_report(sample_y, y_pred))

# PCA - part II
pca_model = PCA(n_components=500)
variance = pca_model.fit(data)
# print(variance)
# print(pca_model.explained_variance_)
# print(pca_model.explained_variance_ratio_)
# print(pca_model.explained_variance_ratio_.cumsum())

pca_data = pca_model.transform(data)