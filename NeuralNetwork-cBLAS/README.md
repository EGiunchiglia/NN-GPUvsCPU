# NeuralNetwork-cBlas

The code is a C++ implementation of a Feedforwad Neural Network with one hidden layer running on a CPU and expoliting cBlas in order to obtain a higher performance (in terms of time). Its initial purposes were: 
1. classyfing digits from the MNIST dataset
2. compare the performances, in terms of time, of a neural network able to run on a GPU exploiting OpenCL and 
   clBlas against a neural network running on a CPU and exploiting cBlas (whose implementation can be found in my other repository
   https://github.com/elenina5/NeuralNetwork-clBLAS).

The neural network takes as input the images of the MNIST dataset, applyes a tanh non linearity and finally classifies them deploying 
the softmax non linearity. The neural network has been trained with the cross-entropy loss and deploying the stocastich gradient descent.
Since the major purpose was to test the 
performances, we make the neural network stop training when either it reaches a 95% accuracy on the training set or 10 epochs have been 
completed. 

The results of the of the performances test can be seen in the following graph: 
![alt text](https://github.com/elenina5/NeuralNetwork-clBLAS/blob/master/performanceCPUvsGPU.png)
