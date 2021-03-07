#### Problem Statement:

A program that implements a decision tree using the ID3 algorithm. Use the following entropy calculation:

![image-20210307190139095](/Users/nandish21/Library/Application Support/typora-user-images/image-20210307190139095.png)

where pi is the proportion of class i (with C being all classes in the data set). Use Information Gain as your decision measure and treat all features as discrete multinomial distributions.

#### Dataset:

car and nursery as csv files. 

#### Steps:

- Read both data sets and treat the last value of each line as the class. 
- Implement the ID3 algorithm and return the final tree without stopping early (both data sets can be learned perfectly, i.e. all leaves have an entropy of 0). 
- The output of the algorithm should look like the example XML solution given for the car data set. With that, you can check the correctness of your solution. 

#### How to run:

Parameters:

1. data - The location of the data file (e.g. /media/data/car.csv).
2. output - Where to write the XML solution to (e.g. /media/data/car solution.xml)

```
         python3 decisiontree.py --data car.csv --output car.xml
```

