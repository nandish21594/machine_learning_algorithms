# Load CSV (using python)
import csv
import numpy
import sys
import argparse

def linearregr():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument('--learningRate', type=float)
    parser.add_argument('--threshold', type=float)
    args=parser.parse_args()
    filename = args.data
    learning_rate = args.learningRate
    threshold = args.threshold
    
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    d = list(reader)
    data = numpy.array(d).astype('float')
    column_len = numpy.size(data, 1)
    x = data[:, 0:column_len - 1]
    y = data[:, column_len - 1: column_len]
    x_with_ones = numpy.c_[numpy.ones(len(x)), x]
    weight = numpy.zeros((x_with_ones.shape[1], 1), float)
    previous_squared_sum_rounded = 0
    i = 0
    # output_file_name = 'solution_yacht_eta0.0001_thres0.0001.csv'
    # with open(output_file_name, mode='w') as f1:
    #     csv_writer = csv.writer(f1, delimiter='\t', lineterminator='\n', )
    while True:
            H = x_with_ones.dot(weight)
            sub = y - H
            gradient = numpy.transpose(x_with_ones).dot(sub)
            squared = numpy.power(sub, 2)
            squared_sum = numpy.sum(squared)
            squared_sum_rounded = numpy.round(squared_sum, 4)
            weight = weight + gradient.dot(learning_rate)
            weight_rounded = numpy.round(weight, 4)  # rounded to 4 decimals
#            print('iteration_number:', i, '\nweights:\n', weight_rounded, '\n sum_of_squared_errors:', squared_sum_rounded,
#                  '\n')
            print(i, ", ".join(str(*x) for x in weight_rounded), squared_sum_rounded)
            
            # csv_data = [i] + [weight_rounded[j] for j in range(weight_rounded.size)] + [squared_sum_rounded]
            # csv_writer.writerow(csv_data)
            if abs(squared_sum_rounded - previous_squared_sum_rounded) < threshold:
                break;
            previous_squared_sum_rounded = squared_sum
            i = i + 1


if __name__ == '__main__':
    linearregr()
