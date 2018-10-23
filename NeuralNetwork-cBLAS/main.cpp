// NOTA: QUESTA INTESTAZIONE MI VIENE AGGIUNTA AUTOMATICAMENTE DALL'EDITOR
//
//  main.cpp
//  sequential_neural_net
//
//  Created by E.Giunchiglia on 03/01/18.
//  Copyright © 2018 Eleonora Giunchiglia. All rights reserved.
//

#include <iostream>
#include <assert.h>
#include "src/dataReader.hpp"
#include "src/neuralNetworkModel.hpp"
#include "src/neuralNetworkTrainer.hpp"


using namespace std;

const int numDatapointsTrainingSet = 60000;
const int numDatapointsTestSet = 10000;
const int lengthDatapoints = 784;
const int numOutputs = 10;

int main(int argc, const char * argv[]) {
    
    assert(argv[1] != "\0");
    openblas_set_num_threads(6);
    
    //Read the number of hidden neurons
    int numHidden = strtol(argv[1], NULL, 10);
    
    if(numHidden < 1){
        cerr << "Number of hidden neurons specified must be > 0 and not " << numHidden << endl;
        return 1;
    }
    
    cout << "Number of hidden neurons: " << numHidden << endl;
    
    float* trainImages = NULL;
    float* trainLabels = NULL;
    float* testImages = NULL;
    float* testLabels = NULL;
    
    readMNISTImages("train", &trainImages);
    readMNISTLabel("train", &trainLabels);
    readMNISTImages("test", &testImages);
    readMNISTLabel("test", &testLabels);
    
    rescaleImages("train", &trainImages, lengthDatapoints);
    rescaleImages("test", &testImages, lengthDatapoints);
    
    float* trainY;
    float* testY;
    createYDataset(trainLabels, &trainY, numOutputs, numDatapointsTrainingSet);
    createYDataset(testLabels, &testY, numOutputs, numDatapointsTestSet);

    
    data* trainDset;
    NetworkTrainer::createDataset(trainImages, trainY, &trainDset, numDatapointsTrainingSet, lengthDatapoints, numOutputs);
    data* testDset;
    NetworkTrainer::createDataset(testImages, testY, &testDset, numDatapointsTestSet, lengthDatapoints, numOutputs);
    
    
    dataset completeDataset{trainDset, testDset};
    

    NetworkModel::Settings networkModelSettings{lengthDatapoints, numHidden, numOutputs};
    
    NetworkModel nn(networkModelSettings);
    
    NetworkTrainer::Settings trainerSettings;
    
    trainerSettings.learningRate = 0.001;
    trainerSettings.maxEpochs = 10;
    trainerSettings.desiredAccuracy = 0.95;
    
    NetworkTrainer nnTrainer(trainerSettings, &nn);
    
    nnTrainer.train(&completeDataset, numDatapointsTrainingSet, numDatapointsTestSet);
    
    return 0;
}

