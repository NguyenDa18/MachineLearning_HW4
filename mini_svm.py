import re
import numpy as np
from sklearn import svm, datasets
from sklearn.decomposition import PCA

lines = []
with open('tester.txt', 'rt') as in_file:
    for line in in_file:
        lines.append(line.rstrip('\n'))

data_list = []
for line in lines:
    label = int(re.split(':', line)[0])
    parsed_input = re.split('1:', line)[1].strip()
    parsed_input = parsed_input.replace("[", "").replace("]", "")
    parsed_input_list = parsed_input.split(',')
    inputs = [float(el) for el in parsed_input_list]
    row = [label] + inputs
    data_list.append(row)
data = np.asarray(data_list)

X = data[:, 1:]
y = data[:, 0]

clf = svm.SVC(kernel='linear', C=1e6)
clf.fit(X, y)
y_pred = clf.predict(X)
print(y_pred)
