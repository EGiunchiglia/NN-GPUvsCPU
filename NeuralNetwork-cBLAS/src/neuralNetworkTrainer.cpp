//
//  neuralNetworkTrainer.cpp
//  sequential_neural_net
//
//  Created by E.Giunchiglia on 03/01/18.
//  Copyright © 2018 Eleonora Giunchiglia. All rights reserved.
//

#include "neuralNetworkTrainer.hpp"
#include <assert.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>

const int CACHELINE = 64;

NetworkTrainer::NetworkTrainer(Settings const& settings, NetworkModel* pNetwork):
pNetwork(pNetwork),
learningRate(settings.learningRate),
desiredAccuracy(settings.desiredAccuracy),
maxEpochs(settings.maxEpochs),
currentEpoch(0),
testSetAccuracy(0)
{
    
    assert(pNetwork != nullptr);
    
    //Allocate memory and initialize them with zeros
    deltaInputHidden = new float[(pNetwork->numInputs+1)*(pNetwork->numHidden+1)+CACHELINE]();
    deltaInputHidden = (float *)(((uintptr_t)deltaInputHidden+CACHELINE)&~(CACHELINE-1));
    deltaHiddenOutput = new float[(pNetwork->numHidden+1)*pNetwork->numOutputs+CACHELINE]();
    deltaHiddenOutput = (float *)(((uintptr_t)deltaHiddenOutput+CACHELINE)&~(CACHELINE-1));
    errorGradientHidden = new float[pNetwork->numHidden+CACHELINE]();
    errorGradientHidden = (float *)(((uintptr_t)errorGradientHidden+CACHELINE)&~(CACHELINE-1));
    errorGradientOutput = new float[pNetwork->numOutputs+CACHELINE]();
    errorGradientOutput = (float *)(((uintptr_t)errorGradientOutput+CACHELINE)&~(CACHELINE-1));
    
}

void NetworkTrainer::train(dataset* data, int numTrainingDatapoints, int numTestDatapoints){
    
    //Reset training state
    currentEpoch=0;
    testSetAccuracy=0;
    float trainSetAccuracy=0;
    
    //Initialize variables necessary to measure time
    chrono::high_resolution_clock::time_point tstart,t1,t2;
    chrono::duration<double> diff;
    double avg = 0.0;
    
    cout << "######################################## " << endl;
    cout << "Start Training" << endl;
    cout << "######################################## " << endl;
    
    while(trainSetAccuracy < desiredAccuracy && currentEpoch < maxEpochs){
    
        //Take time just before the function runEpoch is entered
        t1 = chrono::high_resolution_clock::now();
        //Train the network for one epoch
        runEpoch((data->trainSet), numTrainingDatapoints);
        //Take time just after the function runEpoch is exited
        t2 = chrono::high_resolution_clock::now();
        
        //Sum the time necessary for running an epoch
        diff = t2-t1;
        avg = diff.count();

        
        //Get accuracy on the validation set
        trainSetAccuracy = getAccuracy(data->trainSet, numTrainingDatapoints);
        cout << "Epoch: " + to_string(currentEpoch) + " accuracy on train set: " + to_string(trainSetAccuracy) <<endl;
        
        currentEpoch++;
         
    }
    
    //Take the average time for running an epoch an write it in the .dat file
    avg /= currentEpoch;
    //Open the file for output as text:
    ofstream outfile;
    outfile.open("timePerEpochSeqBLAS.dat", ios_base::app);
    outfile << pNetwork->numHidden << " " << avg << endl;
    outfile.close();
    
    cout << "######################################## " << endl;
    cout << "End training " << endl;
    cout << "######################################## " << endl;
    
    testSetAccuracy = getAccuracy(data->testSet, numTestDatapoints);
    //Get accuracy on the test set
    cout << "Final accuracy on the test set: " << testSetAccuracy << endl;
     
    
}

//Get the error made by the gradient in the node j of the hidden layer
//In order to do so we compute the equation \delta_{j} = (1-tanh^2(z_j)) \sum_{k=1}^M w_{jk} \delta{ouput}
// where z_1 represents the output of the hidden layer before the non-linearity is applied and
// M represents the number of neurons in the output layer

float NetworkTrainer::get1Minustanh2(int hiddenIdx){
    return 2.0/3.0 *(1.7159 - pow(pNetwork->hiddenNeurons[hiddenIdx],2));
}

void NetworkTrainer::computeErrorGradientHidden(){
    float* nonLinearity = new float[pNetwork->numHidden]();
    for(int hiddenIdx=0; hiddenIdx<pNetwork->numHidden; hiddenIdx++)
        nonLinearity[hiddenIdx] = 2.0/3.0 *(1.7159 - pow(pNetwork->hiddenNeurons[hiddenIdx],2));
    
    float* weightedSum = new float[pNetwork->numHidden]();
    cblas_sgemv(CblasRowMajor, CblasNoTrans, pNetwork->numHidden+1, pNetwork->numOutputs, 1.0, pNetwork->weightsHiddenOutput, pNetwork->numOutputs, errorGradientOutput, 1, 0, weightedSum, 1);
    
    for(int hiddenIdx=0; hiddenIdx<pNetwork->numHidden; hiddenIdx++)
        errorGradientHidden[hiddenIdx] = weightedSum[hiddenIdx]*nonLinearity[hiddenIdx];
}


//The function backpropagate is used to compute how much each weight has to change.
//The delta for each weight is computed as the ptoduct of:
//-Learning rate
//-the value of the neuron
//-the error of the gradient in that neuron
void NetworkTrainer::backpropagate(float* trueLabels){
    
    //Get the gradient error for each output node and compute how much each weight has to change
    for(int outputIdx=0; outputIdx<pNetwork->numOutputs; outputIdx++){
        errorGradientOutput[outputIdx] = getErrorGradientOutput(trueLabels[outputIdx], pNetwork->outputNeurons[outputIdx]);
    }
    cblas_sscal(pNetwork->numOutputs*(pNetwork->numHidden+1), 0.01, deltaHiddenOutput,1);
    
    cblas_sger(CblasRowMajor, pNetwork->numHidden+1, pNetwork->numOutputs, learningRate, pNetwork->hiddenNeurons, 1, errorGradientOutput, 1, deltaHiddenOutput, pNetwork->numOutputs);
    
    //Get the gradient error for each hidden node and compute how much each weight has to change
    computeErrorGradientHidden();
    cblas_sscal((pNetwork->numHidden+1)*(pNetwork->numInputs+1), 0.01, deltaInputHidden,1);
    
    cblas_sger(CblasRowMajor, pNetwork->numInputs+1, pNetwork->numHidden+1, learningRate, pNetwork->inputNeurons, 1, errorGradientHidden, 1, deltaInputHidden, pNetwork->numHidden+1);
}

void NetworkTrainer::updateWeights(){
    
    for(int inputIdx=0; inputIdx <=pNetwork->numInputs; inputIdx++){
        for(int hiddenIdx=0; hiddenIdx<=pNetwork->numHidden; hiddenIdx++){
            int weightIdx = pNetwork->getInputHiddenWeightIndex(inputIdx, hiddenIdx);
            pNetwork->weightsInputHidden[weightIdx] -= deltaInputHidden[weightIdx];
        }
    }
    for(int hiddenIdx=0; hiddenIdx <=pNetwork->numHidden; hiddenIdx++){
        for(int outputIdx=0; outputIdx<pNetwork->numOutputs; outputIdx++){
            int weightIdx = pNetwork->getHiddenOutputWeightIndex(hiddenIdx, outputIdx);
            pNetwork->weightsHiddenOutput[weightIdx] -= deltaHiddenOutput[weightIdx];
        }
    }
    
}

void NetworkTrainer::runEpoch(data* trainSet, int numDatapoints){
    
    std::random_shuffle(&trainSet[0], &trainSet[numDatapoints-1]);
    
    for(int i=0; i<numDatapoints; i++){
        pNetwork->evaluate(trainSet[i].X);
        backpropagate(trainSet[i].Y);
        //We use stochastic gradient descent, and thus at each datapoint we update the weights
        updateWeights();
    }
    
}

float NetworkTrainer::getAccuracy(data* testSet, int numDatapoints){
    
    float accuracy = 0;
    int numCorrectResults=0;
    
    
    for(int i=0; i<numDatapoints; i++){
        
        pNetwork->evaluate(testSet[i].X);
        
        bool found = 0;
        float target = -1;
        for(int j=0; j < pNetwork->numOutputs && !found; j++){
            if(testSet[i].Y[j]){
                target=j;
                found=1;
            }
        }
        assert(target>-1);
        if(pNetwork->finalOutput==target)
            numCorrectResults++;
    }
    accuracy = float(numCorrectResults)/numDatapoints;
    return accuracy;
}

void NetworkTrainer::createDataset(float* X, float* Y, data* dataset[], int numDatapoint, int lengthDatapoint, int numOutput){
    
    *dataset = new data[numDatapoint];
    float* xTemp = X;
    float* yTemp = Y;
    for(int i=0; i< numDatapoint; i++){
        (*dataset)[i].X = xTemp+lengthDatapoint*i;
        (*dataset)[i].Y = yTemp+numOutput*i;
    }
}


