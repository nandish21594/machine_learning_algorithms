import csv
import math
import numpy as np
import argparse


def nb():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--output")
    args = parser.parse_args()
    input_file_name = args.data
    output_file_name = args.output
    raw_data = open(input_file_name, 'rt')
    reader = csv.reader(raw_data, delimiter='\t')
    d = list(reader)
    data = np.array(d)
    # setup y data
    data_y = (data[:, 0:1])  # First column is always treated as output y
    for (x, y), value in np.ndenumerate(data_y):
        if data_y[x, y] == 'A':
            data_y[x, y] = 0
        elif data_y[x, y] == 'B':
            data_y[x, y] = 1
    data_y = np.array(data_y).astype('float')
    # setup x data
    data_x = np.array((data[:, 1:]))
    for (x, y), value in np.ndenumerate(data_x):
        if data_x[x, y] == '':
            data_x[x, y] = 0
    data_x = np.array(data_x).astype('float')  # rest of the columns are features
    if data_x.shape[1] > 2 and data_x[0][2] == 0:
        x = np.array(data_x[:, 0: data_x.shape[1] - 1])
    else:
        x = np.array(data_x[:, 0: data_x.shape[1]])
    unique_class_values = np.unique(data_y)
    data_y_unique_counts = np.c_[np.zeros(len(unique_class_values))]
    mu = np.zeros((unique_class_values.size, x.shape[1]))
    sigma = np.zeros((unique_class_values.size, x.shape[1]))
    class_value_probability = np.zeros(unique_class_values.size)
    # calculate mu
    for class_value in range(unique_class_values.size):
        for sample in range(0, data_y.size):
            if data_y[sample] == unique_class_values[class_value]:
                for attr in range(x.shape[1]):
                    mu[class_value][attr] = mu[class_value][attr] + data_x[sample][attr]
                data_y_unique_counts[class_value] = data_y_unique_counts[class_value] + 1
    for attr in range(mu.shape[1]):
        for unique in range(data_y_unique_counts.size):
            mu[unique][attr] = (mu[unique][attr]) / data_y_unique_counts[unique]
    # calculate sigma
    for class_value in range(unique_class_values.size):
        for l in range(0, data_y.size):
            if data_y[l] == unique_class_values[class_value]:
                for attr in range(x.shape[1]):
                    square_term = pow((data_x[l][attr] - mu[class_value][attr]), 2)
                    sigma[class_value][attr] = sigma[class_value][attr] + square_term
    for attr in range(mu.shape[1]):
        for unique in range(data_y_unique_counts.size):
            sigma[unique][attr] = (sigma[unique][attr]) / (data_y_unique_counts[unique] - 1)
    # calculate probability of each class
    for unique in range(data_y_unique_counts.size):
        class_value_probability[unique] = data_y_unique_counts[unique] / sum(data_y_unique_counts)
    # predict the class using individual properties of class which has maximum product of of its attribute probabilities
    attr_probability = np.zeros([x.shape[1], 1])
    class_probabilty = np.zeros([data_y.shape[0], 1])
    predict_class_value = np.zeros(data_y.shape[0])
    for sample in range(0, data_y.size):
        for class_value in range(unique_class_values.size):
            for attr in range(x.shape[1]):
                sample_row = x[sample, :]
                attr_probability[attr] = (1 / math.sqrt(2 * math.pi * sigma[class_value, attr])) * (
                    math.exp(-np.square(sample_row[attr] - mu[class_value, attr]) / (2 * sigma[class_value, attr])))
            class_probabilty[class_value] = np.prod(attr_probability)
        predict_class_value[sample] = np.argmax(class_probabilty)
    # calculate mis-classifications for the data
    misclassified_count = 0
    for sample in range(0, data_y.size):
        if data_y[sample] != predict_class_value[sample]:
            misclassified_count = misclassified_count + 1
    # write the output to tsv file
    with open(output_file_name, mode='w', newline='') as f1:
        csv_writer = csv.writer(f1, delimiter='\t')
        csv_writer.writerow((mu[0, 0], sigma[0, 0], mu[0, 1], sigma[0, 1], class_value_probability[0]))
        csv_writer.writerow((mu[1, 0], sigma[1, 0], mu[1, 1], sigma[1, 1], class_value_probability[1]))
        csv_writer.writerow((misclassified_count, ""))


if __name__ == '__main__':
    nb()
