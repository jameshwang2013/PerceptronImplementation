'''
HOMEWORK9 TEST FILE
'''

import homework9
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
#import random as rand

### TEST BinaryPerceptron()
# Define the training and test data
train = [({"x1": 1}, True), ({"x2": 1}, True), ({"x1": -1}, False), ({"x2": -1}, False)]
test = [{"x1": 1}, {"x1": 1, "x2": 1}, {"x1": -1, "x2": 1.5}, {"x1": -0.5, "x2": -2}]

# Train the classifier for one iteration
p = homework9.BinaryPerceptron(train, 1)

# Make predictions on the test data
print([p.predict(x) for x in test] == [True, True, True, False])


### TEST MulticlassPerceptron()
# Define the training data to be the corners and edge midpoints of the unit square
train = [({"x1": 1}, 1), ({"x1": 1, "x2": 1}, 2), ({"x2": 1}, 3),
         ({"x1": -1, "x2": 1}, 4), ({"x1": -1}, 5), ({"x1": -1, "x2": -1}, 6),
         ({"x2": -1}, 7), ({"x1": 1, "x2": -1}, 8)]

# Train the classifier for 10 iterations so that it can learn each class
p = homework9.MulticlassPerceptron(train, 10)

# Test whether the classifier correctly learned the training data
print([p.predict(x) for x, y in train] == [1, 2, 3, 4, 5, 6, 7, 8])

### TEST IrisClassifier()
# train iris classifier
#accuracy = []
#for i in range(100):

d = homework9.data.iris
#rand.shuffle(d)
#training = d[:120]
#testing = d[120:]

#print(i)
t0 = time.time()
c = homework9.IrisClassifier(d)
t1 = time.time()
print(t1 - t0)
#c_train = homework9.IrisClassifier(training)
#c_test = homework9.IrisClassifier(testing)

# training accuracy
y_hat = [c.classify(x) for x, y in d]
y = [y for x, y in d]
correct = float(sum([1 for i, j in zip(y_hat, y) if i == j]))
#accuracy.append(correct / len(y))
print("Accuracy: {}".format(correct / len(y)))

# test case
print(c.classify((5.1, 3.5, 1.4, 0.2)) == 'iris-setosa')
print(c.classify((7.0, 3.2, 4.7, 1.4)) == 'iris-versicolor')


### TEST DigitClassifier()
d = homework9.data.digits

t0 = time.time()
c = homework9.DigitClassifier(d)
t1 = time.time()
print(t1 - t0)

y_hat = [c.classify(x) for x, y in d]
y = [y for x, y in d]
correct = float(sum([1 for i, j in zip(y_hat, y) if i == j]))
print("Accuracy: {}".format(correct / len(y)))
res = c.classify((0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0))
print(res == 0)

### TEST BiasClassifier
d = homework9.data.bias

t0 = time.time()
c = homework9.BiasClassifier(d)
t1 = time.time()
print(t1 - t0)

y_hat = [c.classify(x) for x, y in d]
y = [y for x, y in d]
correct = float(sum([1 for i, j in zip(y_hat, y) if i == j]))
print("Accuracy: {}".format(correct / len(y)))
print([c.classify(x) for x in (-1, 0, 0.5, 1.5, 2)] == [False, False, False, True, True])


### TEST MysteryClassifier1
import homework9
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

d = homework9.data.mystery1

t0 = time.time()
c = homework9.MysteryClassifier1(d)
t1 = time.time()
print(t1 - t0)

y_hat = [c.classify(x) for x, y in d]
y = [y for x, y in d]
correct = float(sum([1 for i, j in zip(y_hat, y) if i == j]))
print("Accuracy: {}".format(correct / len(y)))
print([c.classify(x) for x in ((0, 0), (0, 1), (-1, 0), (1, 2), (-3, -4))] == [False, False, False, True, True])


# 2D plot the data
vals = np.array([[i[0][0], i[0][1], i[1]] for i in d])
plt.scatter(vals[:, 0], vals[:, 1], c=y_hat)

# 3D plot the data
new_feature = []
for i, j in d:
    x1, x2 = i[0], i[1]
    py = math.sqrt(x1**2 + x2**2)
    if py < 2:
        py = 1
    else:
        py = -1
    new_feature.append(py)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=vals[:, 0], ys=vals[:, 1], zs=new_feature, zdir="z", c=y_hat)
ax.view_init(0, 0)


### TEST MysteryClassifier2
d = homework9.data.mystery2

# train perceptron
t0 = time.time()
c = homework9.MysteryClassifier2(d)
t1 = time.time()
print(t1 - t0)

y_hat = [c.classify(x) for x, y in d]
y = [y for x, y in d]
correct = float(sum([1 for i, j in zip(y_hat, y) if i == j]))
print("Accuracy: {}".format(correct / len(y)))
print([c.classify(x) for x in ((1, 1, 1), (-1, -1, -1), (1, 2, -3), (-1, -2, 3))] == [True, False, False, True])

'''
# plot
vals = np.array([[i[0][0], i[0][1], i[0][2], i[1]] for i in d])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=vals[:, 0], ys=vals[:, 1], zs=vals[:, 2], zdir="y", c=y_hat)
#np.savetxt("plot_3d.csv", vals, delimiter=",") --> visualize in plotly online
'''



