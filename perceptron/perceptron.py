import csv
import numpy
import sys
import argparse


def perceptron():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--output")
    args = parser.parse_args()
    input_file_name = args.data
    output_file_name = args.output
    # input_file_name = "Gauss2.tsv"
    # output_file_name = "Gauss2_wS.tsv"

    raw_data = open(input_file_name, 'rt')
    reader = csv.reader(raw_data, delimiter='\t')
    d = list(reader)
    data = numpy.array(d)
    # setup y data
    data_y = (data[:, 0:1])  # First column is always treated as output y
    for (x, y), value in numpy.ndenumerate(data_y):
        if data_y[x, y] == 'A':
            data_y[x, y] = 1
        elif data_y[x, y] == 'B':
            data_y[x, y] = 0
    data_y = numpy.array(data_y).astype('float')
    # setup x data
    data_x = numpy.array((data[:, 1:]))
    for (x, y), value in numpy.ndenumerate(data_x):
        if data_x[x, y] == '':
            data_x[x, y] = 0
    data_x = numpy.array(data_x).astype('float')  # rest of the columns are features
    data_x_with_ones = numpy.c_[numpy.ones(len(data_x)), data_x]
    print("End-------------")
    misclassified_list = numpy.zeros((2, 101))
    with open(output_file_name, mode='w') as f1:
        csv_writer = csv.writer(f1, delimiter='\t', lineterminator='\n')
        for l in range(1, 3):
            weights = numpy.zeros((data_x_with_ones.shape[1], 1), float)
            mse = []
            print("================================")
            for i in range(1, 102):
                h = data_x_with_ones.dot(weights)
                for (x, y), value in numpy.ndenumerate(h):
                    if h[x, y] > 0:
                        h[x, y] = 1
                    else:
                        h[x, y] = 0
                misclassified_count = 0
                for (x, y), value in numpy.ndenumerate(h):
                    if h[x, y] != data_y[x, y]:
                        misclassified_count = misclassified_count + 1
                mse.append(misclassified_count)
                y_sub_h = data_y - h
                gradient = numpy.transpose(data_x_with_ones).dot(y_sub_h)
                if l == 1:
                    learning_rate = 1
                else:
                    learning_rate = 1 / i
                weights = weights + gradient.dot(learning_rate)
            csv_writer.writerow(mse)


if __name__ == '__main__':
    perceptron()
