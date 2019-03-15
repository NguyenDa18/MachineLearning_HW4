import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn import svm, metrics
from sklearn.cross_validation import train_test_split

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

sample_x = data[0:6, 1:]

# Build the SVM
clf = svm.SVC(kernel='linear')
clf.fit(x, y)

y_pred = clf.predict(sample_x)
print(y_pred)
