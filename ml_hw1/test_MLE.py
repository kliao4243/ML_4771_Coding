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
    train_samples = np.asarray(train_samples, dtype=np.int_)

with open('propublicaTest.csv', 'r') as f:
    reader = csv.reader(f)
    test_samples = list(reader)
    for item in test_samples:
        if len(item) == 0:
            test_samples.remove([])
    test_samples.pop(0)
    test_samples = np.asarray(test_samples, dtype=np.float64)


labels = train_samples[:,0]
train_samples = np.delete(train_samples, 0, 1)
#max = train_samples.max(axis=0)
#train_samples = train_samples/max

positive_samples = []
negative_samples = []
num_pos = 0
num_neg = 0
for i in range(0,len(labels)):
    if labels[i] == 1:
        positive_samples.append(train_samples[i])
        num_pos += 1
    else:
        negative_samples.append(train_samples[i])
        num_neg += 1
pos_rate = num_pos/len(labels)
neg_rate = num_neg/len(labels)

print(num_neg, num_pos, len(labels))


'''
positive_samples = np.transpose(np.asarray(positive_samples, dtype=np.int_))
pos_n_samples = positive_samples.shape[1]
pos_n_features = positive_samples.shape[0]
pos_covariance = np.cov(positive_samples)



pos_inverse_cov = np.linalg.inv(pos_covariance)
pos_det_cov = np.linalg.det(pos_covariance)
pos_mean = np.mean(pos_covariance, axis=1)

'''
neg_det_features = []
var = []
negative_samples = np.transpose(np.asarray(negative_samples, dtype=np.float64))
for i in range(negative_samples.shape[0]-1, -1, -1):
    var.append(np.var(negative_samples[i, :]))

neg_n_samples = negative_samples.shape[1]
neg_n_features = negative_samples.shape[0]
neg_covariance = np.cov(negative_samples)



'''
neg_inverse_cov = np.linalg.inv(neg_covariance)
neg_det_cov = np.linalg.det(neg_covariance)
neg_mean = np.mean(neg_covariance, axis=1)

'''


'''
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

