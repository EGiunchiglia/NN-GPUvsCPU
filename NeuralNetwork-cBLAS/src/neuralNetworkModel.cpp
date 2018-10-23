//
//  neuralNetworkModel.cpp
//  sequential_neural_net
//
//  Created by E.Giunchiglia on 03/01/18.
//  Copyright © 2018 Eleonora Giunchiglia. All rights reserved.
//

#include "neuralNetworkModel.hpp"
#include <assert.h>
#include <random>
#include <cmath>

const int CACHELINE = 64;

//Constructor
NetworkModel::NetworkModel(Settings const& settings)
:numInputs(settings.numInputs),numHidden(settings.numHidden),numOutputs(settings.numOutputs) {
    assert(settings.numInputs > 0 && settings.numHidden > 0 && settings.numOutputs > 0);
    initializeNetwork();
    initializeWeights();
}


//Network initialization
//creates the memory space for neurons and output
//the function initializes all the weights to zero and all the biases to -1.0

void NetworkModel::initializeNetwork() {
    
    //Add bias to the neurons
    int const totNumInputs = numInputs + 1;
    int const totNumHidden = numHidden + 1;
    
    //Compute the total number of weights per layer
    int const numWeightsInputHidden = totNumInputs * totNumHidden;
    int const numWeightsHiddenOutput = totNumHidden * numOutputs;
    
    //Allocate memory for neurons and outputs. Set all to 0.
    inputNeurons = new float[totNumInputs+CACHELINE]();
    inputNeurons = (float *)(((uintptr_t)inputNeurons+CACHELINE)&~(CACHELINE-1));
    hiddenNeurons = new float[totNumHidden+CACHELINE]();
    hiddenNeurons = (float *)(((uintptr_t)hiddenNeurons+CACHELINE)&~(CACHELINE-1));
    outputNeurons = new float[numOutputs+CACHELINE]();
    outputNeurons = (float *)(((uintptr_t)outputNeurons+CACHELINE)&~(CACHELINE-1));
    weightsInputHidden = new float[numWeightsInputHidden+CACHELINE]();
    weightsInputHidden = (float *)(((uintptr_t)weightsInputHidden+CACHELINE)&~(CACHELINE-1));
    weightsHiddenOutput = new float[numWeightsHiddenOutput+CACHELINE]();
    weightsHiddenOutput = (float *)(((uintptr_t)weightsHiddenOutput+CACHELINE)&~(CACHELINE-1));
    
    //Initialize bias values
    inputNeurons[totNumInputs-1] = -1.0;
    hiddenNeurons[totNumHidden-1] = -1.0;
}


//Weights initilization
void NetworkModel::initializeWeights(){
    
    random_device rnd;
    mt19937 generator(rnd());
    
    float const distributionRangeHalfWidth = 2.4/numInputs;
    float const std =  distributionRangeHalfWidth * 2 / 6;
    
    normal_distribution<> nnDstr(0, std);
    
    //Set weights to normally distributed random values between [-1, 1]
    for(int inputIdx = 0; inputIdx <= numInputs; inputIdx++){
        for(int hiddenIdx = 0; hiddenIdx <= numHidden; hiddenIdx++){
            int weightIdx = getInputHiddenWeightIndex(inputIdx, hiddenIdx);
            float weightValue = nnDstr(generator);
            weightsInputHidden[weightIdx] = weightValue;
        }
    }
    for(int hiddenIdx = 0; hiddenIdx <= numHidden; hiddenIdx++){
        for(int outputIdx = 0; outputIdx < numOutputs; outputIdx++){
            int weightIdx = getHiddenOutputWeightIndex(hiddenIdx, outputIdx);
            float weightValue = nnDstr(generator);
            weightsHiddenOutput[weightIdx] = weightValue;
        }
    }
}

//GetOuput
//Given a single input datapoint, get the probabilities for each digit
float* NetworkModel::evaluate(float* input){
    
    assert(inputNeurons[numInputs] == -1.0 && hiddenNeurons[numHidden] == -1.0);
    
    //Set the input values
    cblas_scopy(numInputs, input, 1, inputNeurons, 1);
    
    //CblasRowMajor: it indicates that the matrices are stored in row major order, with the elements of each row of the matrix stored contiguously as shown in the figure above.
    cblas_sgemv(CblasRowMajor, CblasTrans, numInputs+1, numHidden, 1.0, weightsInputHidden, numHidden+1, inputNeurons, 1, 0, hiddenNeurons, 1);
    assert(hiddenNeurons[numHidden] == -1.0);
    
    for (int hiddenIdx = 0; hiddenIdx < numHidden; hiddenIdx++){
        hiddenNeurons[hiddenIdx] = 1.7159*tanh(2.0/3.0 * hiddenNeurons[hiddenIdx]);
    }
    
    cblas_sgemv(CblasRowMajor, CblasTrans, numHidden+1, numOutputs, 1.0, weightsHiddenOutput, numOutputs, hiddenNeurons, 1, 0, outputNeurons, 1);
    
    float softmaxDen = 0;
    for (int outputIdx=0; outputIdx<numOutputs;outputIdx++){
        outputNeurons[outputIdx] = exp(outputNeurons[outputIdx]);
        softmaxDen += (outputNeurons[outputIdx]);
    }
    float max = 0;
    float maxIdx = -1;
    
    for (int outputIdx = 0; outputIdx < numOutputs; outputIdx++){
        outputNeurons[outputIdx] /= softmaxDen;
        if (outputNeurons[outputIdx] >= max){
            max = outputNeurons[outputIdx];
            maxIdx = outputIdx;
        }
    }
    assert(maxIdx > -1);
    finalOutput = maxIdx;
    
    return outputNeurons;
}


