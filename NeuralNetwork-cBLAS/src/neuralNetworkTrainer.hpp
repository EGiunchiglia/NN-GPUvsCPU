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


using namespace std;
struct data{
    float* X;
    float* Y;
};

struct dataset{
    data* trainSet;
    data* testSet;
};

class NetworkTrainer{
    
private:
    
    //Member variables
    NetworkModel* pNetwork;
    float learningRate;
    float desiredAccuracy;
    int maxEpochs;
    
    float* deltaInputHidden;
    float* deltaHiddenOutput;
    float* errorGradientHidden;
    float* errorGradientOutput;
    
    int currentEpoch;
    float testSetAccuracy;
    float validationSetAccuracy;
    float trainCrossEntropy;
    
    //Member functions
    inline float getErrorGradientOutput(float desiredOutput, float output){
        return output-desiredOutput;
    };
    void computeErrorGradientHidden();
    float get1Minustanh2(int hiddenIdx);
    void runEpoch (data* trainSet, int numDatapoints);
    void backpropagate(float* trueLabels);
    void updateWeights();
    
    float getAccuracy(data* trainSet, int numDatapoints);
    
public:
    
    struct Settings {
        float learningRate;
        int maxEpochs;
        float desiredAccuracy;
    };
    
    NetworkTrainer(Settings const& settings, NetworkModel* network);
    void train(dataset* data, int numTrainingDatapoints, int numTestDatapoints);
    static void createDataset(float X[], float Y[], data* dset[], int numDatapoint, int lengthDatapoint, int numOuputs);
};

#endif /* neuralNetworkTrainer_hpp */

