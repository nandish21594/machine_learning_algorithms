**Problem Statement:**

A program that implements a (batch) linear regression using the gradient descent method in Python 3. 

Using the following gradient calculation:

![Screen Shot 2021-03-07 at 12 01 21 PM](https://user-images.githubusercontent.com/20551968/110237552-f2034d00-7f3c-11eb-907b-84be731f70b4.JPG)


**Dataset:**

yacht and random as csv files



**Steps:**

- Read both data sets and treat the last value of each line as the target output. 
- Correctly implement the gradient descent method and return for each iteration the weights and sum of squared errors until a given threshold of change in the error is reached. 
- The output of algorithm will look like this:

```
iteration_number,weight0,weight1,weight2,...,weightN,sum_of_squared_errors
```

- The solution (rounded to 4 decimals) for the random data set is given with a learning rate of 0.0001 and a threshold of 0.0001. With that, we can check the correctness of our solution. 

  

**How to run:**

Parameters:

1. threshold - The threshold, that the change in error has to fall below, before the algorithm terminates.

2. data - The location of the data file (e.g. /media/data/yacht.csv).

3. learningRate - The learning rate of the gradient descent approach. Therefore, I should be able to start your program like this:

```
python3 linearregr.py --data random.csv --learningRate 0.0001 --threshold 0.0001
```

