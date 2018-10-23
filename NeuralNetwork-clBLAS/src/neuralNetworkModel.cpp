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
#include <iostream>

using namespace std;
//Constructor
NetworkModel::NetworkModel(cl_float* trainImages, cl_float* trainY, cl_float* testImages, Settings const& settings, Device* pDevice)
:numInputs(settings.numInputs),numHidden(settings.numHidden),numOutputs(settings.numOutputs),pDevice(pDevice){
    assert(settings.numInputs > 0 && settings.numHidden > 0 && settings.numOutputs > 0);
    
    initializeNetwork();
    initializeWeights();
    
    
    cl_int err = CL_SUCCESS;
    //Create and initialize all buffers a part from the input buffer, which is initialized at each call
    //of the evaluate function
    inputNeurons_bff = clCreateBuffer(pDevice->context, CL_MEM_READ_ONLY, (numInputs+1)*sizeof(cl_float), NULL, &(err));
    hiddenNeurons_bff = clCreateBuffer(pDevice->context, CL_MEM_READ_WRITE  | CL_MEM_COPY_HOST_PTR, ((numHidden+1)*sizeof(cl_float)), hiddenNeurons, &(err));
    if(err != CL_SUCCESS){
        cerr << "error in allocating hidden neurons buffer" << endl;
        exit(1);
    }
    outputNeurons_bff = clCreateBuffer(pDevice->context, CL_MEM_READ_WRITE  | CL_MEM_COPY_HOST_PTR, ((numOutputs)*sizeof(cl_float)), outputNeurons, &(err));
    if(err != CL_SUCCESS){
        cerr << "error in allocating output neurons buffer" << endl;
        exit(1);
    }
    weightsInputHidden_bff = clCreateBuffer(pDevice->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ((numInputs+1)*(numHidden+1)*sizeof(cl_float)), weightsInputHidden, &(err));
    if(err != CL_SUCCESS){
        cerr << "error in allocating  weights input hidden buffer" << endl;
        exit(1);
    }
    weightsHiddenOutput_bff = clCreateBuffer(pDevice->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ((numHidden+1)*(numOutputs)*sizeof(cl_float)), weightsHiddenOutput, &(err));
    if(err != CL_SUCCESS){
        cerr << "error in allocatingweights hidden output buffer" << endl;
        exit(1);
    }
    
    softmaxDen_bff = clCreateBuffer(pDevice->context, CL_MEM_WRITE_ONLY, sizeof(cl_float), NULL, &(err));
    if(err != CL_SUCCESS){
        cerr << "error in allocating softmax denominator buffer" << endl;
        exit(1);
    }
    scratch_bff = clCreateBuffer(pDevice->context, CL_MEM_READ_WRITE, (2*numOutputs*sizeof(cl_float)), NULL, &err);
    if(err != CL_SUCCESS){
        cerr << "error in allocating scratch buffer" << endl;
        exit(1);
    }
    trainY_bff = clCreateBuffer(pDevice->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (60000*numOutputs)*sizeof(cl_float), trainY, &err);
    if(err != CL_SUCCESS){
        cerr << "error in allocating buffer of training Images. Error: " << err << endl;
    }
    trainImages_bff = clCreateBuffer(pDevice->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (60000*numInputs)*sizeof(cl_float), trainImages, &err);
    if(err != CL_SUCCESS){
        cerr << "error in allocating buffer of labels. Error: " << err << endl;
    }
    trueLabel_bff = clCreateBuffer(pDevice->context, CL_MEM_READ_WRITE, (10)*sizeof(cl_float), NULL, &err);
    if(err != CL_SUCCESS){
        cerr << "error in allocating buffer of labels. Error: " << err << endl;
    }
    
    testImages_bff = clCreateBuffer(pDevice->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (10000*numInputs)*sizeof(cl_float), testImages, &err);
    if(err != CL_SUCCESS){
        cerr << "error in allocating buffer of labels. Error: " << err << endl;
    }
}

NetworkModel::~NetworkModel(){
    /* Release OpenCL memory objects. */
    clReleaseMemObject(inputNeurons_bff);
    clReleaseMemObject(hiddenNeurons_bff);
    clReleaseMemObject(outputNeurons_bff);
    clReleaseMemObject(weightsInputHidden_bff);
    clReleaseMemObject(weightsHiddenOutput_bff);
    clReleaseMemObject(softmaxDen_bff);
    clReleaseMemObject(scratch_bff);
    clReleaseMemObject(trainImages_bff);
    clReleaseMemObject(trainY_bff);
    clReleaseMemObject(trueLabel_bff);
    clReleaseMemObject(testImages_bff);    
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
    inputNeurons = new cl_float[totNumInputs]();
    hiddenNeurons = new cl_float[totNumHidden]();
    outputNeurons = new cl_float[numOutputs]();
    weightsInputHidden = new cl_float[numWeightsInputHidden]();
    weightsHiddenOutput = new cl_float[numWeightsHiddenOutput]();
    
    
    //Initialize bias values
    inputNeurons[totNumInputs-1] = -1.0;
    hiddenNeurons[totNumHidden-1] = -1.0;
    
}


//Weights initilization
void NetworkModel::initializeWeights(){
    
    random_device rnd;
    mt19937 generator(rnd());
    
    cl_float const distributionRangeHalfWidth = 2.4/numInputs;
    cl_float const std =  distributionRangeHalfWidth * 2 / 6;
    
    normal_distribution<> nnDstr(0, std);
    
    for(int inputIdx = 0; inputIdx <= numInputs; inputIdx++){
        for(int hiddenIdx = 0; hiddenIdx <= numHidden; hiddenIdx++){
            int weightIdx = getInputHiddenWeightIndex(inputIdx, hiddenIdx);
            cl_float weightValue = nnDstr(generator);
            weightsInputHidden[weightIdx] = weightValue;
        }
    }
    for(int hiddenIdx = 0; hiddenIdx <= numHidden; hiddenIdx++){
        for(int outputIdx = 0; outputIdx < numOutputs; outputIdx++){
            int weightIdx = getHiddenOutputWeightIndex(hiddenIdx, outputIdx);
            cl_float weightValue = nnDstr(generator);
            weightsHiddenOutput[weightIdx] = weightValue;
        }
    }
}

//GetOuput
//Given a single input datapoint, get the probabilities for each digit
cl_float* NetworkModel::evaluate(int numDatum, string phase, string dataset){
    
    assert(inputNeurons[numInputs] == -1.0 && hiddenNeurons[numHidden] == -1.0);
    cl_event event = NULL;
    cl_int err=CL_SUCCESS;
    
    /* Call clblas function. */
    if(dataset.compare("train") == 0){
        err = clblasScopy(numInputs, trainImages_bff, 784*numDatum, 1, inputNeurons_bff, 0, 1, 1, &(pDevice->queue), 0, NULL, &event);
    }
    else if (dataset.compare("test") == 0){
         err = clblasScopy(784, testImages_bff, 784*numDatum, 1, inputNeurons_bff, 0, 1, 1, &(pDevice->queue), 0, NULL, &event);
    }
    else{
        cerr << "Dataset specified in evaluate function wrong. Either specify phase train or tes" << endl;
        exit(1);
    }
    if (err != CL_SUCCESS) {
        printf("clblasScopy() of input data failed with %d\n", err);
        exit(1);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Release OpenCL events. */
        clReleaseEvent(event);
    }
    
    /* Call clblas extended function. Perform multiplication input and hidden layer*/
    err = clblasSgemv(clblasRowMajor, clblasTrans, numInputs+1, numHidden, 1.0,
                      weightsInputHidden_bff, 0, numHidden+1, inputNeurons_bff, 0, 1, 0,
                      hiddenNeurons_bff, 0, 1, 1, &(pDevice->queue), 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSgemvEx() multiplying input per hidden failed with %d\n", err);
        exit(1);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        clReleaseEvent(event);
    }
    
    
    size_t global_pattern[] = {(size_t)numHidden+1, 0,0};
    
    
    err |= clSetKernelArg(pDevice->tanh_kernel, 0, sizeof(hiddenNeurons_bff), &hiddenNeurons_bff);
    if (err != CL_SUCCESS){
        cerr << "error in creating or passing parameters to kernel. Error: " << err << endl;
        exit(1);
    }
    err = clEnqueueNDRangeKernel(pDevice->queue, pDevice->tanh_kernel, 1, NULL, global_pattern, pDevice->group_pattern, 0, NULL, &event);
    if(err != CL_SUCCESS){
        cerr << "error in getting result. Error:" << err << endl;
        exit(1);
    }else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        clReleaseEvent(event);
    }
    
    
    /* Release OpenCL events, kernel and program */
    err = clblasSgemv(clblasRowMajor, clblasTrans, numHidden+1, numOutputs, 1.0,
                      weightsHiddenOutput_bff, 0, numOutputs, hiddenNeurons_bff, 0, 1, 0,
                      outputNeurons_bff, 0, 1, 1, &(pDevice->queue), 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSgemvEx() multiplying hidden per output failed with %d\n", err);
        exit(1);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Release OpenCL events. */
        clReleaseEvent(event);
    }
    
    size_t global_output_pattern[] = {(size_t)numOutputs, 0,0};
    
    err |= clSetKernelArg(pDevice->exp_kernel, 0, sizeof(outputNeurons_bff), &outputNeurons_bff);
    if (err != CL_SUCCESS){
        cerr << "error in creating or passing parameters to kernel. Error: " << err << endl;
        exit(1);
    }
    err = clEnqueueNDRangeKernel(pDevice->queue, pDevice->exp_kernel, 1, NULL, global_output_pattern, pDevice->group_pattern, 0, NULL, &event);
    if(err != CL_SUCCESS){
        cerr << "error in getting result. Error:" << err << endl;
        exit(1);
    }else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        /* Fetch results of calculations from GPU memory. */
        //err = clEnqueueReadBuffer(pDevice->queue, outputNeurons_bff, CL_TRUE, 0, numOutputs * sizeof(*outputNeurons), outputNeurons, 0, NULL, NULL);
        clReleaseEvent(event);
    }
    
    /* Call clblas function. */
    err = clblasSasum(numOutputs, softmaxDen_bff, 0, outputNeurons_bff, 0, 1, scratch_bff,
                      1, &(pDevice->queue), 0, NULL, &event);
    
    if (err != CL_SUCCESS) {
        printf("clblasSasum() failed with %d\n", err);
        exit(1);
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        clReleaseEvent(event);

    }
    
    err |= clSetKernelArg(pDevice->multiplyConstant_kernel, 0, sizeof(outputNeurons_bff), &outputNeurons_bff);
    err |= clSetKernelArg(pDevice->multiplyConstant_kernel, 1, sizeof(softmaxDen_bff), &softmaxDen_bff);
    if (err != CL_SUCCESS){
        cerr << "error in creating or passing parameters to kernel. Error: " << err << endl;
        exit(1);
    }
    err = clEnqueueNDRangeKernel(pDevice->queue, pDevice->multiplyConstant_kernel, 1, NULL, global_output_pattern, pDevice->group_pattern, 0, NULL, &event);
    if(err != CL_SUCCESS){
        cerr << "error in getting result. Error:" << err << endl;
        exit(1);
    }else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);
        clReleaseEvent(event);
    }
    
    if(phase.compare("testing") == 0){
        
        err = clEnqueueReadBuffer(pDevice->queue, outputNeurons_bff, CL_TRUE, 0, numOutputs * sizeof(*outputNeurons), outputNeurons, 0, NULL, NULL);
        cl_float maxIdx = -1;
        cl_float max = 0;

        for (int outputIdx = 0; outputIdx < numOutputs; outputIdx++){
             if (outputNeurons[outputIdx] >= max){
                 max = outputNeurons[outputIdx];
                 maxIdx = outputIdx;
             }
         }
        assert(maxIdx > -1);
        finalOutput = maxIdx;
    }
    
    return outputNeurons;
}



