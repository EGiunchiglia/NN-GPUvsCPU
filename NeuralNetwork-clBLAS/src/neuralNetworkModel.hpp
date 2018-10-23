//
//  neuralNetworkModel.hpp
//  sequential_neural_net
//
//  Created by E.Giunchiglia on 03/01/18.
//  Copyright © 2018 Eleonora Giunchiglia. All rights reserved.
//

#ifndef neuralNetworkModel_hpp
#define neuralNetworkModel_hpp

#include <stdio.h>
#include <vector>
#include <math.h>
#include <clBLAS.h>
#include <string>
#include "Device.hpp"



using namespace std;


class NetworkModel {
    
    friend class NetworkTrainer;
    friend class Device;
    
private:
    
    int numInputs;
    int numHidden;
    int numOutputs;
    
    cl_float* inputNeurons;
    cl_float* hiddenNeurons;
    cl_float* outputNeurons;
    
    cl_float finalOutput;
    
    cl_float* weightsInputHidden;
    cl_float* weightsHiddenOutput;
    
    cl_mem inputNeurons_bff;
    cl_mem hiddenNeurons_bff;
    cl_mem outputNeurons_bff;
    
    cl_mem weightsInputHidden_bff;
    cl_mem weightsHiddenOutput_bff;
    
    cl_mem softmaxDen_bff;
    cl_mem scratch_bff;
    
    cl_mem trainImages_bff;
    cl_mem trainY_bff;
    cl_mem trueLabel_bff;
    
    cl_mem testImages_bff;
    
    Device* pDevice;
    
    void initializeNetwork();
    void initializeWeights();
    
    inline int getInputHiddenWeightIndex(int inputIdx, int hiddenIdx){
        //Here we need +1 because we added the bias neuron in the hidden layer
        return inputIdx * (numHidden+1) + hiddenIdx;
    }
    inline int getHiddenOutputWeightIndex(int hiddenIdx, int outputIdx){
        return hiddenIdx * (numOutputs) + outputIdx;
    }
    
    inline static cl_float tanhFunction(cl_float x){
        return tanh(x);
    }
    
    inline static cl_float sigmoidFunction(cl_float x){
        return 1.0/ (1.0+exp(-x));
    }
    
public:
    
    struct Settings{
        int numInputs;
        int numHidden;
        int numOutputs;
    };
    
    NetworkModel(cl_float* trainImages, cl_float* trainY, cl_float* testImages, Settings const& settings, Device* pDevice);
    ~NetworkModel();
    
    cl_float* evaluate(int numDatum, string phase, string dataset);
};

#endif /* neuralNetworkModel_hpp */


