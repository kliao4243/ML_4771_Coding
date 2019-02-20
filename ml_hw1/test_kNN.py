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


def knn_classify(x_train, test_sample, k, k_norm):
    dis_matrix = x_train - test_sample
    max = np.absolute(dis_matrix).max(axis=0)
    dis_matrix = dis_matrix / max
    distance = np.linalg.norm(dis_matrix, k_norm, axis=1)
    neighbor = [0] * k
    for i in range(0, distance.shape[0]):
        for j in range(0, k):
            if distance[i] < distance[neighbor[j]]:
                neighbor = neighbor[0:j] + [i] + neighbor[j:k - 1]
                break
    #print(neighbor)
    positive_count = 0
    negative_count = 0
    for i in range(0, k):
        if train_labels[neighbor[i]] == 1:
            positive_count += 1
        else:
            negative_count += 1
    if positive_count > negative_count:
        return 1
    else:
        return 0


if __name__ == "__main__":

    # knn accuracy
    correct_count = 0
    n_neighbor = int(input("Input K value:"))
    d_norm = int(input("Input degree of norm:"))
    for i in range(0, test_samples.shape[0]):
        if knn_classify(train_samples, test_samples[i], n_neighbor, d_norm) == test_labels[i]:
            correct_count += 1
    print(correct_count/test_samples.shape[0])

