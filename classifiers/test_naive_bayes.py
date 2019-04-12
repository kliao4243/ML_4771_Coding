import numpy as np
import csv

# pre-process

with open('propublicaTrain.csv', 'r') as f:
    reader = csv.reader(f)
    train_samples = list(reader)
    for item in train_samples:
        if len(item) == 0:
            train_samples.remove([])
    train_samples.pop(0)
    train_samples = np.asarray(train_samples, dtype=np.float64)
    train_labels = train_samples[:, 0]
    train_samples = np.delete(train_samples, 0, 1)
    train_samples = np.delete(train_samples, 2, 1)


with open('propublicaTest.csv', 'r') as f:
    reader = csv.reader(f)
    test_samples = list(reader)
    for item in test_samples:
        if len(item) == 0:
            test_samples.remove([])
    test_samples.pop(0)
    test_samples = np.asarray(test_samples, dtype=np.float64)
    test_labels = test_samples[:, 0]
    test_samples = np.delete(test_samples, 0, 1)
    test_samples = np.delete(test_samples, 2, 1)



class NBClassifier(object):
    def __init__(self, x_train, y_labels):
        self.positive_samples = []
        self.negative_samples = []
        num_pos = 0
        num_neg = 0
        for i in range(0, len(y_labels)):
            if y_labels[i] == 1:
                self.positive_samples.append(x_train[i])
                num_pos += 1
            else:
                self.negative_samples.append(x_train[i])
                num_neg += 1
        self.pos_rate = num_pos / len(train_labels)
        self.neg_rate = num_neg / len(train_labels)
        self.positive_samples = np.asarray(self.positive_samples, dtype=np.float64)
        self.negative_samples = np.asarray(self.negative_samples, dtype=np.float64)
        self.positive_probability = list()
        self.negative_probability = list()

        for i in range(0, self.positive_samples.shape[1]):
            temp_feature = self.positive_samples[:, i]
            temp_prob = dict()
            for feature in temp_feature:
                if str(feature) in temp_prob:
                    temp_prob[str(feature)] += 1 / self.positive_samples.shape[0]
                else:
                    temp_prob[str(feature)] = 1 / self.positive_samples.shape[0]
            self.positive_probability.append(temp_prob)
        for i in range(0, self.negative_samples.shape[1]):
            temp_feature = self.negative_samples[:, i]
            temp_prob = dict()
            for feature in temp_feature:
                if str(feature) in temp_prob:
                    temp_prob[str(feature)] += 1 / self.negative_samples.shape[0]
                else:
                    temp_prob[str(feature)] = 1 / self.negative_samples.shape[0]
            self.negative_probability.append(temp_prob)

    def classify(self, x_test):
        test_pos_prob = self.pos_rate
        for i in range(0, self.positive_samples.shape[1]):
            try:
                test_pos_prob *= self.positive_probability[i][str(x_test[i])]
            except:
                test_pos_prob *= 0.5/self.positive_samples.shape[0]
        test_neg_prob = self.neg_rate
        for i in range(0, self.negative_samples.shape[1]):
            try:
                test_neg_prob *= self.negative_probability[i][str(x_test[i])]
            except:
                test_neg_prob *= 0.5/self.negative_samples.shape[0]

        #print(test_pos_prob, test_neg_prob)
        if test_pos_prob > test_neg_prob:
            return 1
        else:
            return 0


if __name__ == "__main__":
    clf = NBClassifier(train_samples, train_labels)
    correct_count = 0
    for i in range(0, test_samples.shape[0]):
        if clf.classify(test_samples[i]) == test_labels[i]:
            correct_count += 1
    print(correct_count / test_samples.shape[0])



'''
positive_samples = []
negative_samples = []
num_pos = 0
num_neg = 0
for i in range(0,len(train_labels)):
    if train_labels[i] == 1:
        positive_samples.append(train_samples[i])
        num_pos += 1
    else:
        negative_samples.append(train_samples[i])
        num_neg += 1
pos_rate = num_pos/len(train_labels)
neg_rate = num_neg/len(train_labels)
positive_samples = np.asarray(positive_samples, dtype=np.float64)
negative_samples = np.asarray(negative_samples, dtype=np.float64)


positive_probability = list()

for i in range(0, positive_samples.shape[1]):
    temp_feature = positive_samples[:, i]
    temp_prob = dict()
    for feature in temp_feature:
        if str(feature) in temp_prob:
            temp_prob[str(feature)] += 1/positive_samples.shape[0]
        else:
            temp_prob[str(feature)] = 1/positive_samples.shape[0]
    positive_probability.append(temp_prob)

negative_probability = list()

for i in range(0, negative_samples.shape[1]):
    temp_feature = negative_samples[:, i]
    temp_prob = dict()
    for feature in temp_feature:
        if str(feature) in temp_prob:
            temp_prob[str(feature)] += 1/negative_samples.shape[0]
        else:
            temp_prob[str(feature)] = 1/negative_samples.shape[0]
    negative_probability.append(temp_prob)


test_pos_prob = pos_rate

for i in range(0, positive_samples.shape[1]):
    test_pos_prob *= positive_probability[i][str(test_data[i])]

test_neg_prob = neg_rate

for i in range(0, negative_samples.shape[1]):
    test_neg_prob *= negative_probability[i][str(test_data[i])]

print(test_pos_prob, test_neg_prob)
'''