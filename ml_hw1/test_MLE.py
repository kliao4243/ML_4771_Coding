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


class MLEClassifier(object):
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
        self.positive_samples = np.transpose(np.asarray(self.positive_samples, dtype=np.float64))
        self.negative_samples = np.transpose(np.asarray(self.negative_samples, dtype=np.float64))
        self.pos_rate = num_pos / len(y_labels)
        self.neg_rate = num_neg / len(y_labels)
        pos_covariance = np.cov(self.positive_samples) + pow(10, -3) * np.identity(8)
        self.pos_inverse_cov = np.linalg.inv(pos_covariance)
        pos_det_cov = np.linalg.det(pos_covariance)
        self.pos_mean = np.mean(pos_covariance, axis=1)
        self.pos_cons = 1/(np.sqrt(pow((2*np.pi), x_train.shape[1])*pos_det_cov))
        neg_covariance = np.cov(self.negative_samples) + pow(10, -3) * np.identity(8)
        self.neg_inverse_cov = np.linalg.inv(neg_covariance)
        neg_det_cov = np.linalg.det(neg_covariance)
        self.neg_mean = np.mean(neg_covariance, axis=1)
        self.neg_cons = 1 / (np.sqrt(pow((2 * np.pi), x_train.shape[1]) * neg_det_cov))

    def classify(self, x_test):
        test_pos_prob = self.pos_rate * self.pos_cons * np.exp((-0.5) * \
                            np.dot(np.dot(np.transpose(x_test-self.pos_mean),\
                                          self.pos_inverse_cov), (x_test-self.pos_mean)))
        test_neg_prob = self.neg_rate * self.neg_cons * np.exp((-0.5)*\
                            np.dot(np.dot(np.transpose(x_test-self.neg_mean),\
                                          self.neg_inverse_cov), (x_test-self.neg_mean)))
        #print(test_neg_prob, test_pos_prob)
        if test_pos_prob > test_neg_prob:
            return 1
        else:
            return 0


if __name__ == "__main__":
    clf = MLEClassifier(train_samples, train_labels)
    clf.classify(test_samples[10])
    correct_count = 0
    for i in range(0, test_samples.shape[0]):
        if clf.classify(test_samples[i]) == test_labels[i]:
            correct_count += 1
    print(correct_count / test_samples.shape[0])

'''
positive_samples = np.transpose(np.asarray(positive_samples, dtype=np.int_))
pos_n_samples = positive_samples.shape[1]
pos_n_features = positive_samples.shape[0]
pos_covariance = np.cov(positive_samples)



pos_inverse_cov = np.linalg.inv(pos_covariance)
pos_det_cov = np.linalg.det(pos_covariance)
pos_mean = np.mean(pos_covariance, axis=1)


neg_det_features = []
var = []
negative_samples = np.transpose(np.asarray(negative_samples, dtype=np.float64))
for i in range(negative_samples.shape[0]-1, -1, -1):
    var.append(np.var(negative_samples[i, :]))

neg_n_samples = negative_samples.shape[1]
neg_n_features = negative_samples.shape[0]
neg_covariance = np.cov(negative_samples)
print(neg_covariance)







class MLClassifier ( object ):

    def __init__ ( self, x_train):
        self.x_label = x_train[:, 0]
        self.x_features = np.delete(x_train, 0, 1)
        self.x_features = np.transpose(self.x_features)
        self.n_samples = self.x_features.shape[1]
        self.n_features = self.x_features.shape[0]
        self.covariance = np.cov(self.x_features)
        self.inverse_cov = np.linalg.inv(self.covariance)
        self.determinant_cov = np.linalg.det(self.covariance)
        self.mean = np.mean(self.x_features, axis=1)
'''

