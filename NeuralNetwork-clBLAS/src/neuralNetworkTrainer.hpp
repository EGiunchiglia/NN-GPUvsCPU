//
//  neuralNetworkTrainer.hpp
//  sequential_neural_net
//
//  Created by E.Giunchiglia on 03/01/18.
//  Copyright © 2018 Eleonora Giunchiglia. All rights reserved.
//

#ifndef neuralNetworkTrainer_hpp
#define neuralNetworkTrainer_hpp

#include <stdio.h>
#include <vector>
#include "neuralNetworkModel.hpp"
#include <clBLAS.h>


using namespace std;
struct data{
    cl_float* X;
    cl_float* Y;
};

struct dataset{
    data* trainSet;
    data* testSet;
};

class NetworkTrainer{
    
private:
    
    //Member variables
    NetworkModel* pNetwork;
    
    cl_float learningRate;
    cl_float desiredAccuracy;
    int maxEpochs;
    
    cl_float* deltaInputHidden;
    cl_float* deltaHiddenOutput;
    cl_float* errorGradientHidden;
    cl_float* errorGradientOutput;
    
    cl_mem deltaInputHidden_bff;
    cl_mem deltaHiddenOutput_bff;
    cl_mem errorGradientHidden_bff;
    cl_mem errorGradientOutput_bff;
    cl_mem nonLinearity_bff;
    cl_mem weightedSum_bff;
    
    int currentEpoch;
    cl_float testSetAccuracy;
    cl_float validationSetAccuracy;
    cl_float trainCrossEntropy;
    
    //Member functions
    inline cl_float getErrorGradientOutput(cl_float desiredOutput, cl_float output){
        return output-desiredOutput;
    };
    void computeErrorGradientHidden();
    cl_float get1Minustanh2(int hiddenIdx);
    void runEpoch (int numDatapoints);
    void backpropagate(int numDatum);
    void updateWeights();
    
    cl_float getAccuracy(data* trainSet, int numDatapoints, string dataset);
    
public:
    
    struct Settings {
        cl_float learningRate;
        int maxEpochs;
        cl_float desiredAccuracy;
    };
    
    NetworkTrainer(Settings const& settings, NetworkModel* network);
    ~NetworkTrainer();
    void train(dataset* data, int numTrainingDatapoints, int numTestDatapoints);
    static void createDataset(cl_float X[], cl_float Y[], data* dset[], int numDatapoint, int lengthDatapoint, int numOuputs);
};

#endif /* neuralNetworkTrainer_hpp */


