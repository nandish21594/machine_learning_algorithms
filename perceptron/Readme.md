#### Problem Statement:

A program that implements a single perceptron using the below delta rule. Use the following activation function:



where w⃗ is the vector of weights including the bias (w0). Treat all attributes and weights as double-precision values.



#### Dataset:

Example and Gauss2 as tsv (tabular separated values)

#### Steps:

- Read both data sets and treat the first value of each line as the class (A or B). 
- In order to get the same results, class A is to be treated as the positive class, hence y = 1, and class B as the negative one (y = 0). 
- All weights are to be initialized with 0. 
- Implement the perceptron learning rule in batch mode with a constant (ηt = η0) and an annealing (ηt = η0 ) learning rate (in both cases η0 = 1), i.e:



​		where Y(⃗x,w⃗) is the set of samples which are misclassified.

- The number of misclassified points is the error rate (i.e. |Y(⃗x,w⃗)|).

- The output of the algorithm is a single tsv file, which contains exactly two rows after 100 iterations (per variant):

  1. The first row contains the tabular separated values for the error of each iteration (starting from iteration 0) with the constant learning rate.
  2. The second row follows the same format, but with the annealing learning rate.

- 

  

  How to run:

Parameters:

1. data - The location of the data file (e.g. /media/data/car.csv). 
2. output - Where the output tsv should be written to.

```
         python3 perceptron.py --data Example.tsv --output Example_Errors.tsv
```

#### How to run:

The figures below shows the data for the Example set and its single perceptron solution after 100 iterations with a constant learning rate.