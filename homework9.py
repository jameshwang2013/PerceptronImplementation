############################################################
# CIS 521: Homework 9
############################################################

student_name = "James Wang"

############################################################
# Imports
############################################################

import homework9_data as data

# Include your imports here, if any are used.

from collections import defaultdict
from collections import OrderedDict
import math

############################################################
# Section 1: Perceptrons
############################################################

class BinaryPerceptron(object):

    def __init__(self, examples, iterations):

        # create dictionary of features to labels
        self.features = OrderedDict.fromkeys(sum([i[0].keys() for i in examples], [])).keys()
        X = defaultdict()
        for i in examples:
            obs = []
            for j in self.features:
                try:
                    x = i[0][j]
                except:
                    x = 0
                obs.append(x)
            X[tuple(obs)] = i[1]

        # train perceptron algorithm
        self.w = [0] * len(self.features)
        for i in range(iterations):
            for j in X:

                x_i = list(j)
                y_i = X[j]
                y_i_hat = sum([i * j for i, j in zip(self.w, x_i)]) > 0

                if y_i_hat != y_i:

                    if y_i_hat is True:
                        self.w = [i - j for i, j in zip(self.w, x_i)]
                    else:
                        self.w = [i + j for i, j in zip(self.w, x_i)]

    def predict(self, x):

        obs = []
        for j in self.features:
            try:
                x_i = x[j]
            except:
                x_i = 0
            obs.append(x_i)

        return sum([i * j for i, j in zip(self.w, obs)]) > 0


class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):

        # create dictionary of features to labels
        self.features = OrderedDict.fromkeys(sum([i[0].keys() for i in examples], [])).keys()
        X = OrderedDict()
        for i in examples:
            obs = []
            #for j in features:
            for j in self.features:
                try:
                    x = i[0][j]
                except:
                    x = 0
                obs.append(x)
            X[tuple(obs)] = i[1]
        labels = X.values()

        # train perceptron algorithm
        self.w_dict = {i: [0] * len(self.features) for i in labels}
        for i in range(iterations):
            for j in X:
                x_i = list(j)
                y_i = X[j]
                y_i_hat_dict = {wgt: sum([i * j for i, j in zip(self.w_dict[wgt], x_i)])
                                for wgt in self.w_dict}
                y_i_hat = max(y_i_hat_dict, key=y_i_hat_dict.get)

                if y_i_hat != y_i:
                    self.w_dict[y_i] = [m + n for m, n in zip(self.w_dict[y_i], x_i)]
                    self.w_dict[y_i_hat] = [m - n for m, n in zip(self.w_dict[y_i_hat], x_i)]

    def predict(self, x):

        obs = []
        for j in self.features:
            try:
                x_i = x[j]
            except:
                x_i = 0
            obs.append(x_i)
        scores = {wgt: sum([i * j for i, j in zip(self.w_dict[wgt], obs)])
                  for wgt in self.w_dict}

        return max(scores, key=scores.get)


############################################################
# Section 2: Applications
############################################################

class IrisClassifier(object):

    def __init__(self, data):
        list_data = [([("x{}".format(j), i[0][j]) for j in range(len(i[0]))], i[1])
                     for i in data]
        self.iris_data = [(OrderedDict(i), j) for i, j in list_data]
        self.clf = MulticlassPerceptron(self.iris_data, 25)

    def classify(self, instance):
        new = [("x{}".format(j), instance[j]) for j in range(len(instance))]
        return self.clf.predict(OrderedDict(new))


class DigitClassifier(object):

    def __init__(self, data):
        list_data = [([("x{}".format(j), i[0][j]) for j in range(len(i[0]))], i[1])
                     for i in data]
        self.digit_data = [(OrderedDict(i), j) for i, j in list_data]
        self.clf = MulticlassPerceptron(self.digit_data, 10)

    def classify(self, instance):
        new = [("x{}".format(j), instance[j]) for j in range(len(instance))]
        return self.clf.predict(OrderedDict(new))


class BiasClassifier(object):

    def __init__(self, data):
        kern = [((i[0], i[0]-1, i[0]**2), i[1]) for i in data]
        list_data = [([("x{}".format(j), i[0][j]) for j in range(len(i[0]))], i[1])
                     for i in kern]
        self.bias_data = [(OrderedDict(i), j) for i, j in list_data]
        self.clf = BinaryPerceptron(self.bias_data, 3)

    def classify(self, instance):
        kern = (instance, instance - 1, instance**2)
        new = [("x{}".format(j), kern[j]) for j in range(len(kern))]
        return self.clf.predict(OrderedDict(new))


class MysteryClassifier1(object):

    def __init__(self, data):
        kern = []
        for i, j in data:
            x, y, z = i[0], i[1], math.sqrt(i[0]**2 + i[1]**2)
            if z < 2:
                z = 1
            else:
                z = -1
            kern.append(((x, y, z), j))
        list_data = [([("x{}".format(j), i[0][j]) for j in range(len(i[0]))], i[1])
                     for i in kern]
        self.mystery1_data = [(OrderedDict(i), j) for i, j in list_data]
        self.clf = BinaryPerceptron(self.mystery1_data, 1)

    def classify(self, instance):
        z = math.sqrt(instance[0]**2 + instance[1]**2)
        if z < 2:
            z = 1
        else:
            z = -1
        kern = (instance[0], instance[1], z)
        new = [("x{}".format(j), kern[j]) for j in range(len(kern))]
        return self.clf.predict(OrderedDict(new))


class MysteryClassifier2(object):

    def __init__(self, data):
        kern = []
        for i, j in data:
            x, y, z = i[0], i[1], i[2]
            if x > 0 and y > 0 and z > 0:
                xyz = 1
            else:
                if x < 0 and y < 0 and z > 1:
                    xyz = 1
                else:
                    if x < 0 and y > 0 and z < 0:
                        xyz = 1
                    else:
                        if x > 0 and y < 0 and z < 0:
                            xyz = 1
                        else:
                            xyz = -1
            kern.append(((x, y, z, xyz), j))

        list_data = [([("x{}".format(j), i[0][j]) for j in range(len(i[0]))], i[1])
                     for i in kern]
        self.mystery2_data = [(OrderedDict(i), j) for i, j in list_data]
        self.clf = BinaryPerceptron(self.mystery2_data, 10)

    def classify(self, instance):
        x, y, z = instance[0], instance[1], instance[2]
        if x > 0 and y > 0 and z > 0:
            xyz = 1
        else:
            if x < 0 and y < 0 and z > 1:
                xyz = 1
            else:
                if x < 0 and y > 0 and z < 0:
                    xyz = 1
                else:
                    if x > 0 and y < 0 and z < 0:
                        xyz = 1
                    else:
                        xyz = -1

        kern = (x, y, z, xyz)
        new = [("x{}".format(j), kern[j]) for j in range(len(kern))]
        return self.clf.predict(OrderedDict(new))


############################################################
# Section 3: Feedback
############################################################

feedback_question_1 = """
15 hours
"""

feedback_question_2 = """
Tuning the number of iterations parameter was a bit challenging. With no test
set, I was constantly worried that I was overfitting the data by minimizing
the training misclassification rate. I ended up splitting up the data into
training and testing sets (70%/30% split) to evaluate my classifers on data
it never saw before.
"""

feedback_question_3 = """
It was interesting to implement the perceptron algorithm without numpy. It
made me really critically think about the algorithm in a way I had not before.
I also found these applications very interesting and fun.
"""
