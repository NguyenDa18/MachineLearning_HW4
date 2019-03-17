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

sample_x = data[:, 1:]
sample_y = y[:]

# Build the SVM
clf = svm.SVC(kernel='linear')
clf.fit(x, y)
y_pred = clf.predict(sample_x)

count = 0
for label, predicted in zip(sample_y, y_pred):
    if (label == predicted):
        continue
    print("%r" % label == predicted)
    count += 1
print(count)

scores = zip(sample_y, y_pred)
mapped_scores = list(scores)

# Model Evaluation
print("---- SVM MODEL REPORT -----")
print("Accuracy: %f" % accuracy_score(sample_y, y_pred))
print(classification_report(sample_y, y_pred))

# PCA - part II
pca_model = PCA(n_components=100)
variance = pca_model.fit(data)
print("--- PCA ---")
print(variance)
print("Eigenvalues: \n")
print(pca_model.explained_variance_)
print("Explained variance ratio")
print(pca_model.explained_variance_ratio_)
print("Explained variance cum. sum")
print(pca_model.explained_variance_ratio_.cumsum())
print("Principal components: \n")
print(pca_model.components_)

pca_data = pca_model.transform(data)
print(pca_data)