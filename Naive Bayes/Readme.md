#### Problem Statement:

A program that implements a 2-class Naive Bayes algorithm with an apriori decision rule using a multinomial estimation for the classes and a gaussian estimation for the attributes.  The formulas to be used are below:



where xa is an instance x with an attribute a and μ and σ being the parameters of the Gaussian. The parameter estimates are given as follows:



where nci is the amount of instances for class ci.

#### Dataset:

Example and Gauss2 as tsv (tabular separated values)

#### Steps:

- Read both data sets and treat the first value of each line as the class (A or B). 
- The output of your algorithm should be a single tsv file per data set, which contains a row for each class:



- The last (third) row contains the absolute number of misclassifications for the data. Any other information should not be inside the output file, only the requested values.

Parameters:

1. data - The location of the data file (e.g. /media/data/Example.tsv). 
2. output - Where the output tsv should be written to.

#### How to run:

```
          python3 nb.py --data Example.tsv --output Example_NB_Solution.tsv
```

#### 

The figures below shows the data for the Example set and its Naive Bayes solution.