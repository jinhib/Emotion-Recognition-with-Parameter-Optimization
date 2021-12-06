import numpy as np
import csv
import math
from scipy import stats


def point_biserial_coef(feature, label):
    # correlation_coefficient[0] = correlation
    # correlation_coefficient[1] = p-value

    feature = np.array(feature)
    label = np.array(label)

    feature = feature.astype('float64')
    label = label.astype('float64')

    correlation_coefficient = stats.pointbiserialr(feature, label)

    coef = correlation_coefficient[0]

    if math.isnan(coef):
        return 0

    return coef ** 2


def pearson_coef(feature, label):
    # correlation_coefficient[0] = correlation
    # correlation_coefficient[1] = p-value

    feature = np.array(feature)
    label = np.array(label)

    feature = feature.astype('float64')
    label = label.astype('float64')

    correlation_coefficient = stats.pearsonr(feature, label)

    coef = correlation_coefficient[0]

    if math.isnan(coef):
        return 0

    return abs(coef)


for num in range(1, 21):
    dataset = np.genfromtxt('face_distance_features_balance.csv', delimiter=',', encoding='UTF8')
    dataset = dataset[1:]
    print('len_of_original_features : ', len(dataset[0]) - 1)

    dataset = np.array(dataset)

    scores = []
    for index in range(len(dataset[0]) - 1):
        score = pearson_coef(dataset[:, index], dataset[:, -1])
        temp = [score, index]
        scores.append(temp)

    sorted_list = sorted(scores, key=lambda x: x[0], reverse=True)
    sorted_list = np.array(sorted_list)
    pearson_rank_index = sorted_list[:, 1]

    feature_name = np.genfromtxt('face_distance_features_balance.csv', delimiter=',', encoding='UTF8', dtype=str)
    feature_name = feature_name[0]
    temp = []

    original_dataset = np.genfromtxt('face_distance_features_balance.csv', delimiter=',', encoding='UTF8', skip_header=1)
    labels = original_dataset[:, -1]
    labels = np.expand_dims(labels, axis=1)
    dataset = original_dataset[:, :-1]
    dataset = np.transpose(dataset)
    feature_name = np.transpose(feature_name)

    subset = []
    f_name = []
    for index in pearson_rank_index[:num]:
        subset.append(dataset[int(index)])
        f_name.append(feature_name[int(index)])

    f_name.append('label')
    print('len_of_pearson_features : ', len(subset))

    subset = np.transpose(subset)
    f_name = np.transpose(f_name)
    f_name = np.expand_dims(f_name, axis=0)

    dataset = np.concatenate((subset, labels), axis=1)
    fisher_dataset = np.concatenate((f_name, dataset), axis=0)

    f = open('pearson_' + str(len(subset[0])) + '_selected_face_distance_features_balance.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for line in fisher_dataset:
        wr.writerow(line)
    f.close()
