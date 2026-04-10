# FNNs
Using Feedfoward Neural Networks to perform sentiment analysis on IMDB movie reviews

## Problem 1

### To reload data

- run `python preprocessing.py`

## Problem 2, 3

### To test the already generated baseline model

- run `python baseline.py`

###  To regenerate the baseline model

- Uncomment line 191 in the main function of baseline.py (`# lr_wd_test()`) 
- Run `python baseline.py`

## Problem 4

### To train and test k-fold cross validation implementation

- Run `python kfold_train.py`

## Problem 5

### To train and test both single FNN with dropout and the bagging implementation

- Run `python dropout_train.py`