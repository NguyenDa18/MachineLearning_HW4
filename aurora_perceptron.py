import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

mu = input("Enter a learning rate (0.1, 0.01, 0.001, 1.5): ")
epochs = input("Enter number of epochs: ")
batch_size = input("Enter a batch size (up to 5699): ")

mu = float(mu)
epochs = int(epochs)
batch_size = int(batch_size)

plt.figure(figsize=(10,8))
plt.subplots_adjust(hspace=.5)
plt.subplot(211)

plt.title("Aurora perceptron learning, " + ' Batch Size: ' + str(batch_size) + ', ' + str(epochs) + ' epochs' + ', Mu: ' + str(mu))
plt.xlabel("X1")
plt.ylabel("X2")

np.random.seed(0)

# Read in data
lines_data = []
with open('new.txt', 'rt') as in_file:
    for line in in_file:
        lines_data.append(line.rstrip('\n'))

data_list = []
for line in lines_data:
    label = int(re.split(':', line)[0])

    parsed_input = re.split('1:', line)[1].strip()
    parsed_input = parsed_input.replace("[", "")
    parsed_input = parsed_input.replace("]", "")
    parsed_input_list = parsed_input.split(',')
    inputs = [float(el) for el in parsed_input_list]
    row = [label] + inputs
    data_list.append(row)
data = np.asarray(data_list)

L = data[:,0] # labels of samples

# Generate random weights of 256 + 256 + 256 AND 1 for initial bias
W = np.random.random_sample((256 * 3) + 1,) - np.random.random_sample()
# W = np.array([0, 1, 0.5])


rows = data.shape[0] # 5699
cols = data.shape[1] # 769 or 256 + 256 + 256 + 1

############################################################
                    ## PERCEPTRON ##
############################################################
total_accuracy = []
current_accuracy = 0
for j in range(epochs):
    accuracy = 0
    total_accuracy.append(current_accuracy)
    for i in range(batch_size):
        charge = W[0] + np.dot(data[i, 1:], W[1:])
        print('i: %d' % i)
        predict = 1 if charge > 0 else 0

        if predict == L[i]:
            accuracy += 1
        else:
            Error = predict - L[i]
            W_t = W
            X_t  = np.concatenate(([1], data[i,1:]))
            W_t = np.multiply(mu, np.multiply(Error, X_t))
            W = np.subtract(W, W_t)
            print("Error: %f charge: %f predict: %f L[i]: %f" % (Error, charge, predict, L[i]))
            
            plt.plot (np.arange (-2.5,2.5,0.1), -W[0]/W[-1] - W[1]/W[-1] * np.arange(-2.5,2.5,0.1))
    print("Epoch: %d" % j)
    current_accuracy = float(accuracy) / batch_size
    print("Accuracy: %f" % (float(accuracy) / batch_size))

plt.subplot(212)
plt.title('Classification Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

for count, value in enumerate(total_accuracy):
    print("Epoch: %d, Accuracy: %f" % (count+1, value))
    plt.scatter(count + 1, value)
plt.show()