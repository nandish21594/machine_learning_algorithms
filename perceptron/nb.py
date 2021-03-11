#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 21:41:00 2019

@author: priyankabyahatti
"""
import numpy as np
import pandas as pd
import argparse
import tsv
from math import sqrt
from math import pi
from math import exp

# parser = argparse.ArgumentParser()
# parser.add_argument("--data")
# parser.add_argument('--output')
# args=parser.parse_args()
#filename = args.data
filename = "/Users/nandish21/Downloads/1st_Semester/ML/ProgrammingAssignment/Example.tsv"
#Read data, Set Iteration
mydata = pd.read_csv(filename, sep='\t', header=None)

## Data Preprocessing
N = len(mydata)
Ncol = len(mydata.columns)
##mydata = mydata.iloc[:,0: Ncol-1]
mydata[0] = (mydata[0].replace('A', 1))
mydata[0] = (mydata[0].replace('B', 0))
mydata = pd.DataFrame(mydata)
mydata = mydata[[1, 2, 0]]

##Calculating Prior Probabilities
n_outcome1 = mydata[0][mydata[0] == 1].count()
n_outcome0 = mydata[0][mydata[0] == 0].count()
n_total = mydata[0].count()
P_outcome1 = n_outcome1/n_total
P_outcome0 = n_outcome0/n_total

##Separate data by class variable
mydata_0 = mydata[mydata[0] == 0]
mydata_1 = mydata[mydata[0] == 1]

##Defining function to calculate mean
def mean(numbers):
    return sum(numbers)/float(len(numbers))

##Defining function to calculate standard deviation
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return sqrt(variance)

##Defining function to calculate standard deviation
def variance(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return variance

# Calculate the Gaussian probability distribution function for x input
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


## Calculating mean and std for all attributes with respect to classes
mean_x1_1 = np.mean(mydata_1[1])
mean_x2_1 = np.mean(mydata_1[2])
mean_x1_0 = np.mean(mydata_0[1])
mean_x2_0 = np.mean(mydata_0[2])        

print(mean_x1_1)
print(mean_x1_0)
print(mean_x2_1)
print(mean_x2_0)

std_x1_1 = stdev(mydata_1[1])
std_x2_1 = stdev(mydata_1[2])
std_x1_0 = stdev(mydata_0[1])
std_x2_0 = stdev(mydata_0[2])

var_x1_1 = variance(mydata_1[1])
var_x2_1 = variance(mydata_1[2])
var_x1_0 = variance(mydata_0[1])
var_x2_0 = variance(mydata_0[2])    


##Naive Bayes Classifier
dataset = mydata
probabilities = pd.DataFrame(columns = ["prob1", "prob0"])
for i in range(len(dataset)):
    prob1 = calculate_probability(dataset[1][i], mean_x1_1, std_x1_1)*calculate_probability(dataset[2][i], mean_x2_1, std_x2_1)*P_outcome1
    prob0 = calculate_probability(dataset[1][i], mean_x1_0, std_x1_0)*calculate_probability(dataset[2][i], mean_x2_0, std_x2_0)*P_outcome0
    probabilities = probabilities.append({'prob1': prob1, 'prob0': prob0}, ignore_index = True)
result_data  = pd.concat([mydata.reset_index(drop=True), probabilities], axis=1)
result_data["pred_class"] = 0

##assign class to instance with maximum probability value
for i in range(len(result_data)):
    if (result_data["prob1"].iloc[i] > result_data["prob0"].iloc[i]):
        result_data["pred_class"].iloc[i] = 1
    else: result_data["pred_class"].iloc[i] = 0

##Calculate misclassification with training data
miss_c = np.where((result_data[0] == result_data["pred_class"]), 1, 0)
no_of_misclassifications = np.count_nonzero(miss_c == 0)

list1 =   [mean_x1_1, var_x1_1, mean_x2_1, var_x2_1, P_outcome1]
list2 =   [mean_x1_0, var_x1_0, mean_x2_0, var_x2_0, P_outcome0]
list3 =   [no_of_misclassifications]

print(list1)
print(list2)
print(list3)

#save output as tab delimited file
#writer = tsv.TsvWriter(open(args.output, "w"))
#writer.list_line(list1)
#writer.list_line(list2)
#writer.list_line(list3)
#writer.close()