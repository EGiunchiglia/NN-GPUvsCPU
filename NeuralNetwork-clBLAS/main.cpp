
//
//  main.cpp
//  sequential_neural_net
//
//  Created by E.Giunchiglia on 03/01/18.
//  Copyright © 2018 Eleonora Giunchiglia. All rights reserved.
//

#include <iostream>
#include <stdlib.h>
#include <assert.h>


#include "src/dataReader.hpp"
#include "src/neuralNetworkModel.hpp"
#include "src/neuralNetworkTrainer.hpp"
#include "src/Device.cpp"
 
#include <clBLAS.h>

#define MAX_SOURCE_SIZE (0x100000)

using namespace std;

const int numDatapointsTrainingSet = 60000;
const int numDatapointsTestSet = 10000;
const int lengthDatapoints = 784;
const int numOutputs = 10;

int main(int argc, const char * argv[]) {
    
    assert(argv[1] != '\0');
    
    //Read the number of hidden neurons
    int numHidden = strtol(argv[1], NULL, 10);
    
    if(numHidden < 1){
        cerr << "Number of hidden neurons specified must be > 0 and not " << numHidden << endl;
        return 1;
    }
    
    cout << "Number of hidden neurons: " << numHidden << endl;
    
    
    //////
    // READ THE KERNEL FILES
    //////
    // Load the source code containing the kernel
    FILE *fp;
    char fileName[] = "./src/kernel.cl";
    char *source_str;
    size_t tanh_size;
    
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    cout << "Start to read file" << endl;
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    tanh_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    cout << "Read kernel file" << endl;
    
    
    
    //////
    // IMPORT THE DATASET
    /////
    cl_float* trainImages = NULL;
    cl_float* trainLabels = NULL;
    cl_float* testImages = NULL;
    cl_float* testLabels = NULL;
    
    readMNISTImages("train", &trainImages);
    readMNISTLabel("train", &trainLabels);
    readMNISTImages("test", &testImages);
    readMNISTLabel("test", &testLabels);
    
    rescaleImages("train", &trainImages, lengthDatapoints);
    rescaleImages("test", &testImages, lengthDatapoints);
    
    
    cl_float* trainY;
    cl_float* testY;
    createYDataset(trainLabels, &trainY, 10, numDatapointsTrainingSet);
    createYDataset(testLabels, &testY, 10, numDatapointsTestSet);
    
    
    data* trainDset;
    NetworkTrainer::createDataset(trainImages, trainY, &trainDset, numDatapointsTrainingSet, lengthDatapoints, numOutputs);
    data* testDset;
    NetworkTrainer::createDataset(testImages, testY, &testDset, numDatapointsTestSet, lengthDatapoints, numOutputs);
    
    
    dataset completeDataset{trainDset, testDset};
    Device device(source_str);
    
    
    
    NetworkModel::Settings networkModelSettings{lengthDatapoints, numHidden, numOutputs};
    NetworkModel nn(trainImages, trainY, testImages, networkModelSettings, &device);
    
    
    NetworkTrainer::Settings trainerSettings;
    trainerSettings.learningRate = 0.001;
    trainerSettings.maxEpochs = 10;
    trainerSettings.desiredAccuracy = 0.95;
    NetworkTrainer nnTrainer(trainerSettings, &nn);
    
    nnTrainer.train(&completeDataset, numDatapointsTrainingSet, numDatapointsTestSet);
    
    
    return 0;
}


