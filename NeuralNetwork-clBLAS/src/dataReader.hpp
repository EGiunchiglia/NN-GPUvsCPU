//
//  dataReader.hpp
//  sequential_neural_net
//
//  Created by E.Giunchiglia on 03/01/18.
//  Copyright © 2018 Eleonora Giunchiglia. All rights reserved.
//

#ifndef dataReader_hpp
#define dataReader_hpp

#include <clBLAS.h>
#include <stdio.h>
#include <vector>
#include <string>

int ReverseInt (int i);
void readMNISTImages(std::string type,cl_float* arr[]);
void readMNISTLabel(std::string type, cl_float* vec[]);

void rescaleImages(std::string type, cl_float* arr[], int lengthDatapoint);

void createYDataset(cl_float* testLabel, cl_float* y[], int numOutput, int numImages);

#endif /* dataReader_hpp */


