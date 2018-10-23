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

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>    /* The MacOS X blas/lapack stuff */
#else
#include <cblas.h>              /* C BLAS          BLAS  */
#endif


using namespace std;


class NetworkModel {
    
    friend class NetworkTrainer;

public:
    float* outputNeurons;
private:
    
    int numInputs;
    int numHidden;
    int numOutputs;
    
    float* inputNeurons;
    float* hiddenNeurons;
    //float* outputNeurons;
    
    float finalOutput;
    
    float* weightsInputHidden;
    float* weightsHiddenOutput;
    
    
    void initializeNetwork();
    void initializeWeights();
    
    inline int getInputHiddenWeightIndex(int inputIdx, int hiddenIdx){
        //Here we need +1 because we added the bias neuron in the hidden layer
        return inputIdx * (numHidden+1) + hiddenIdx;
    }
    inline int getHiddenOutputWeightIndex(int hiddenIdx, int outputIdx){
        return hiddenIdx * (numOutputs) + outputIdx;
    }
    
    inline static float tanhFunction(float x){
        return tanh(x);
    }
    
    inline static float sigmoidFunction(float x){
        return 1.0/ (1.0+exp(-x));
    }
    
public:
    
    struct Settings{
        int numInputs;
        int numHidden;
        int numOutputs;
    };
    
    NetworkModel(Settings const& settings);
    
    float* evaluate(float* input);
    float* getWeightsInputHidden() {return weightsInputHidden;}
    float* getWeightsHiddenOutput() {return weightsHiddenOutput;}
};

#endif /* neuralNetworkModel_hpp */

