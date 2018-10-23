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
#include <string>


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
    deltaInputHidden = new cl_float[(pNetwork->numInputs+1)*(pNetwork->numHidden+1)]();
    deltaHiddenOutput = new cl_float[(pNetwork->numHidden+1)*pNetwork->numOutputs]();
    errorGradientHidden = new cl_float[pNetwork->numHidden]();
    errorGradientOutput = new cl_float[pNetwork->numOutputs]();
    
    //Initialize buffers
    cl_int err = CL_SUCCESS;
    deltaInputHidden_bff = clCreateBuffer(pNetwork->pDevice->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ((pNetwork->numInputs+1)*(pNetwork->numHidden+1) * sizeof(cl_float)), deltaInputHidden, &err);
    if(err != CL_SUCCESS){
        cerr << "error in allocating deltaInputHidden buffer. Error: " << err << endl;
        exit(1);
    }
    deltaHiddenOutput_bff = clCreateBuffer(pNetwork->pDevice->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ((pNetwork->numHidden+1)*pNetwork->numOutputs) * sizeof(cl_float), deltaHiddenOutput, &err);
    if(err != CL_SUCCESS){
        cerr << "error in allocating deltaHiddenOutput buffer. Error: " << err << endl;
        exit(1);
    }
    errorGradientHidden_bff = clCreateBuffer(pNetwork->pDevice->context, CL_MEM_READ_WRITE, (pNetwork->numHidden+1)* sizeof(cl_float), NULL, &err);
    if(err != CL_SUCCESS){
        cerr << "error in allocating errorGradientHidden buffer. Error: " << err << endl;
        exit(1);
    }
    errorGradientOutput_bff = clCreateBuffer(pNetwork->pDevice->context, CL_MEM_READ_WRITE, (pNetwork->numOutputs)*sizeof(cl_float), NULL, &err);
    if(err != CL_SUCCESS){
        cerr << "error in allocating errorGradientOutput buffer. Error: " << err << endl;
        exit(1);
    }
    nonLinearity_bff = clCreateBuffer(pNetwork->pDevice->context, CL_MEM_READ_WRITE, pNetwork->numHidden*sizeof(cl_float), NULL, &err);
    if(err != CL_SUCCESS){
        cerr << "error in allocating non linearity buffer. Error: " << err << endl;
    }
    weightedSum_bff = clCreateBuffer(pNetwork->pDevice->context, CL_MEM_READ_WRITE, (pNetwork->numHidden+1)*sizeof(cl_float), NULL, &err);
    if(err != CL_SUCCESS){
        cerr << "error in allocating weightedSum buffer. Error: " << err << endl;
    }
}

NetworkTrainer :: ~NetworkTrainer(){
    clReleaseMemObject(deltaInputHidden_bff);
    clReleaseMemObject(deltaHiddenOutput_bff);
    clReleaseMemObject(errorGradientHidden_bff);
    clReleaseMemObject(errorGradientOutput_bff);
    clReleaseMemObject(nonLinearity_bff);
    clReleaseMemObject(weightedSum_bff);
}

void NetworkTrainer::train(dataset* data, int numTrainingDatapoints, int numTestDatapoints){
    
    cl_int err = CL_SUCCESS;
    
    chrono::high_resolution_clock::time_point tstart,t1,t2;
    chrono::duration<double> diff;
    double avg = 0.0;
    
    //Reset training state
    currentEpoch=0;
    testSetAccuracy=0;
    cl_float trainSetAccuracy=0;
    
    cout << "######################################## " << endl;
    cout << "Start Training" << endl;
    cout << "######################################## " << endl;
    
    while(trainSetAccuracy < desiredAccuracy && currentEpoch < maxEpochs){
        
        //Take time just before the function runEpoch is entered
        t1 = chrono::high_resolution_clock::now();
        //Train the network for one epoch
        runEpoch(numTrainingDatapoints);
        //Take time just after the function runEpoch is exited
        t2 = chrono::high_resolution_clock::now();
        
        //Sum the time necessary for running an epoch
        diff = t2-t1;
        avg = diff.count();
        
        err = clEnqueueReadBuffer(pNetwork->pDevice->queue, pNetwork->outputNeurons_bff, CL_TRUE, 0, pNetwork->numOutputs * sizeof(*pNetwork->outputNeurons), pNetwork->outputNeurons, 0, NULL, NULL);
        
        //Get accuracy on the train set
        trainSetAccuracy = getAccuracy(data->trainSet, numTrainingDatapoints, "train");
        cout << "Epoch: " + to_string(currentEpoch) + " accuracy on train set: " + to_string(trainSetAccuracy) <<endl;
        
        currentEpoch++;
        
    }
    //Take the average time for running an epoch an write it in the .dat file
    avg /= currentEpoch;
    //Open the file for output as text:
    ofstream outfile;
    outfile.open("timePerEpochGPU.dat", ios_base::app);
    outfile << pNetwork->numHidden << " " << avg << endl;
    outfile.close();
    
    cout << "######################################## " << endl;
    cout << "End training " << endl;
    cout << "######################################## " << endl;
    
    //Get accuracy on the test set
    testSetAccuracy = getAccuracy(data->testSet, numTestDatapoints, "test");
    cout << "Epoch: " + to_string(currentEpoch) + " accuracy on test set: " + to_string(testSetAccuracy) <<endl;
    cout << "Final accuracy on the test set: " << testSetAccuracy << endl;
}

//Get the error made by the gradient in the node j of the hidden layer
//In order to do so we compute the equation \delta_{j} = (1-tanh^2(z_j)) \sum_{k=1}^M w_{jk} \delta{ouput}
// where z_1 represents the output of the hidden layer before the non-linearity is applied and
// M represents the number of neurons in the output layer
void NetworkTrainer::computeErrorGradientHidden(){
    cl_float* nonLinearity = new cl_float[pNetwork->numHidden]();
    
    //Create buffer for storing the result
    cl_int err = CL_SUCCESS;
    cl_event event = NULL;
    
    size_t global_pattern[] = {(size_t)pNetwork->numHidden, 0,0};
    
    err |= clSetKernelArg(pNetwork->pDevice->nonLinearity_kernel, 0, sizeof(nonLinearity_bff), &nonLinearity_bff);
    err |= clSetKernelArg(pNetwork->pDevice->nonLinearity_kernel, 1, sizeof(pNetwork->hiddenNeurons_bff), &pNetwork->hiddenNeurons_bff);
    if (err != CL_SUCCESS){
        cerr << "error in creating or passing parameters to kernel for creating gradient hidden non linearity. Error: " << err << endl;
        exit(1);
    }
    
    err = clEnqueueNDRangeKernel(pNetwork->pDevice->queue, pNetwork->pDevice->nonLinearity_kernel, 1, NULL, global_pattern, pNetwork->pDevice->group_pattern, 0, NULL, &event);
    if(err != CL_SUCCESS){
        cerr << "error in getting result. Error:" << err << endl;
        exit(1);
    }else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Release OpenCL events, kernel and program */
        clReleaseEvent(event);
    }
    
    err = clblasSgemv(clblasRowMajor, clblasNoTrans, pNetwork->numHidden+1, pNetwork->numOutputs, 1.0,
                      pNetwork->weightsHiddenOutput_bff, 0, pNetwork->numOutputs, errorGradientOutput_bff, 0, 1.0, 0, weightedSum_bff, 0, 1, 1, &(pNetwork->pDevice->queue), 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("in network trainer: clblasSgemvEx() failed with %d\n", err);
        exit(1);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Release OpenCL events, kernel and program */
        clReleaseEvent(event);
    }
   
    err |= clSetKernelArg(pNetwork->pDevice->mul_kernel, 0, sizeof(errorGradientHidden_bff), &errorGradientHidden_bff);
    err |= clSetKernelArg(pNetwork->pDevice->mul_kernel, 1, sizeof(nonLinearity_bff), &nonLinearity_bff);
    err |= clSetKernelArg(pNetwork->pDevice->mul_kernel, 2, sizeof(weightedSum_bff), &weightedSum_bff);
    
    if (err != CL_SUCCESS){
        cerr << "error in creating or passing parameters to kernel. Error: " << err << endl;
        exit(1);
    }
    
    err = clEnqueueNDRangeKernel(pNetwork->pDevice->queue, pNetwork->pDevice->mul_kernel, 1, NULL, global_pattern, pNetwork->pDevice->group_pattern, 0, NULL, &event);
    if(err != CL_SUCCESS){
        cerr << "error in getting result. Error:" << err << endl;
        exit(1);
    }else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Release OpenCL events, kernel and program */
        clReleaseEvent(event);
    }
}


//The function backpropagate is used to compute how much each weight has to change.
//The delta for each weight is computed as the product of:
//-Learning rate
//-the value of the neuron
//-the error of the gradient in that neuron
void NetworkTrainer::backpropagate(int numDatum){
    
    cl_int err = CL_SUCCESS;
    cl_event event = NULL;
    /* Call clblas function. */
    err = clblasScopy(10, pNetwork->trainY_bff, 10*numDatum, 1, pNetwork->trueLabel_bff, 0, 1, 1, &(pNetwork->pDevice->queue), 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasScopy() failed with %d\n", err);
        exit(1);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Release OpenCL events. */
        clReleaseEvent(event);
    }
    size_t global_pattern_input_hidden[] = {(size_t)(pNetwork->numOutputs), 0,0};
    
    //Get the gradient error for each output node and compute how much each weight has to change
    err |= clSetKernelArg(pNetwork->pDevice->elementwiseSubtraction_kernel, 0, sizeof(errorGradientOutput_bff), &errorGradientOutput_bff);
    err |= clSetKernelArg(pNetwork->pDevice->elementwiseSubtraction_kernel, 1, sizeof(pNetwork->outputNeurons_bff), &pNetwork->outputNeurons_bff);
    err |= clSetKernelArg(pNetwork->pDevice->elementwiseSubtraction_kernel, 2, sizeof(pNetwork->trueLabel_bff), &pNetwork->trueLabel_bff);
    if (err != CL_SUCCESS){
        cerr << "error in creating or passing parameters to kernel for updating weights. Error: " << err << endl;
        exit(1);
    }
    err = clEnqueueNDRangeKernel(pNetwork->pDevice->queue, pNetwork->pDevice->elementwiseSubtraction_kernel, 1, NULL, global_pattern_input_hidden, pNetwork->pDevice->group_pattern, 0, NULL, &event);
    if(err != CL_SUCCESS){
        cerr << "error in getting result. Error:" << err << endl;
        exit(1);
    }else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Release OpenCL events, kernel and program */
        clReleaseEvent(event);
    }
    
    /* Call clblas function. */
    err = clblasSscal(pNetwork->numOutputs*(pNetwork->numHidden+1), 0.01, deltaHiddenOutput_bff, 0, 1, 1, &(pNetwork->pDevice->queue), 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSscal() failed with %d\n", err);
        exit(1);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Release OpenCL events, kernel and program */
        clReleaseEvent(event);
    }
    
    err = clblasSger(clblasRowMajor, pNetwork->numHidden+1, pNetwork->numOutputs, learningRate, pNetwork->hiddenNeurons_bff, 0, 1, errorGradientOutput_bff, 0, 1, deltaHiddenOutput_bff, 0, pNetwork->numOutputs, 1, &(pNetwork->pDevice->queue), 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSger() failed with %d\n", err);
        exit(1);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Release OpenCL events. */
        clReleaseEvent(event);
    }
    //Get the gradient error for each hidden node and compute how much each weight has to change
    computeErrorGradientHidden();
    
    /* Call clblas function. */
    err = clblasSscal((pNetwork->numHidden+1)*(pNetwork->numInputs+1), 0.01, deltaInputHidden_bff, 0, 1, 1, &(pNetwork->pDevice->queue), 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSscal() failed with %d\n", err);
        exit(1);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Release OpenCL events. */
        clReleaseEvent(event);
    }
    err = clblasSger(clblasRowMajor, pNetwork->numInputs+1, pNetwork->numHidden, learningRate, pNetwork->inputNeurons_bff, 0, 1, errorGradientHidden_bff, 0, 1, deltaInputHidden_bff, 0, pNetwork->numHidden+1, 1, &(pNetwork->pDevice->queue), 0, NULL, &event);
    
    if (err != CL_SUCCESS) {
        printf("clblasSger() failed with %d\n", err);
        exit(1);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Release OpenCL events. */
        clReleaseEvent(event);
    }
    
}
//The function update weights is used to update the value of the weights once the partial derivatives
//are computed.
void NetworkTrainer::updateWeights(){
    
    cl_int err;
    cl_event event = NULL;
    
    size_t global_pattern_input_hidden[] = {(size_t)(pNetwork->numInputs+1)*(pNetwork->numHidden+1), 0,0};
    
    err |= clSetKernelArg(pNetwork->pDevice->subtract_kernel, 0, sizeof(pNetwork->weightsInputHidden_bff), &pNetwork->weightsInputHidden_bff);
    err |= clSetKernelArg(pNetwork->pDevice->subtract_kernel, 1, sizeof(deltaInputHidden_bff), &deltaInputHidden_bff);
    if (err != CL_SUCCESS){
        cerr << "error in creating or passing parameters to kernel for updating weights. Error: " << err << endl;
        exit(1);
    }
    err = clEnqueueNDRangeKernel(pNetwork->pDevice->queue, pNetwork->pDevice->subtract_kernel, 1, NULL, global_pattern_input_hidden, pNetwork->pDevice->group_pattern, 0, NULL, &event);
    if(err != CL_SUCCESS){
        cerr << "error in getting result. Error:" << err << endl;
        exit(1);
    }else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Release OpenCL events, kernel and program */
        clReleaseEvent(event);
    }
    
    size_t global_pattern_hidden_output[] = {(size_t)((pNetwork->numHidden+1)*(pNetwork->numOutputs)), 0, 0};
    err |= clSetKernelArg(pNetwork->pDevice->subtract_kernel, 0, sizeof(pNetwork->weightsHiddenOutput_bff), &pNetwork->weightsHiddenOutput_bff);
    err |= clSetKernelArg(pNetwork->pDevice->subtract_kernel, 1, sizeof(deltaHiddenOutput_bff), &deltaHiddenOutput_bff);
    if (err != CL_SUCCESS){
        cerr << "error in creating or passing parameters to kernel for updating weights. Error: " << err << endl;
        exit(1);
    }
    
    err = clEnqueueNDRangeKernel(pNetwork->pDevice->queue, pNetwork->pDevice->subtract_kernel, 1, NULL, global_pattern_hidden_output, pNetwork->pDevice->group_pattern, 0, NULL, &event);
    if(err != CL_SUCCESS){
        cerr << "error in getting result. Error:" << err << endl;
        exit(1);
    }else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Release OpenCL events, kernel and program */
        clReleaseEvent(event);
    }
}

//The function runEpoch inputs once each datapoint of the dataset in the neural network,
//makes both the backward and the backward pass and on the ground of these it updates the weights.
void NetworkTrainer::runEpoch(int numDatapoints){
    
    for(int i=0; i<numDatapoints; i++){
        pNetwork->evaluate(i,  "training", "train");
        backpropagate(i);
        //We use stochastic gradient descent, and thus at each datapoint we update the weights
        updateWeights();
    }
}


//The function getAccuracy computes the accuracy deploying the field finalOutput of the class NetworkModel.
cl_float NetworkTrainer::getAccuracy(data* testSet, int numDatapoints, string dataset){
    
    cl_float accuracy = 0;
    int numCorrectResults=0;
    
    
    for(int i=0; i<numDatapoints; i++){
        
        pNetwork->evaluate(i, "testing", dataset);

        bool found = 0;
        cl_float target = -1;
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
    accuracy = cl_float(numCorrectResults)/numDatapoints;
     return accuracy; 
}

//The function createDataset allows us to create a list of data, namely a list of couples <datapoint, label>
void NetworkTrainer::createDataset(cl_float* X, cl_float* Y, data* dataset[], int numDatapoint, int lengthDatapoint, int numOutput){
    
    *dataset = new data[numDatapoint];
    cl_float* xTemp = X;
    cl_float* yTemp = Y;
    for(int i=0; i< numDatapoint; i++){
        (*dataset)[i].X = xTemp+lengthDatapoint*i;
        (*dataset)[i].Y = yTemp+numOutput*i;
    }
}



