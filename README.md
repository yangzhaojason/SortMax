## Project Introduction

1. For the detection of normal data and abnormal behavior data, the industry-used Isolation Forest unsupervised algorithm is used, and the accuracy of the experimental measurement is over 90%;
2. Classify the user behavior data, use the supervised algorithm to use the neural network, compare the convolutional neural network (CNN) used at the beginning, and finally consider the timing of the user's APP behavior. We use the long-term and short-term memory neural network ( LSTM), the classification accuracy is above 96%, and the entire data volume reaches GB level or above;

##  Code running

1、Use the KDD CUP99 data set to modify the .csv file path and run 'python iforest_test.py'

2、Then run 'python lstm_keras.py'

## image display

Result.jpeg is a demonstration of the results. The accuracy of multiple classifications is as high as 99% and above.