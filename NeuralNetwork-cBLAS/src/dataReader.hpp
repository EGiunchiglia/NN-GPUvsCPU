//
//  dataReader.hpp
//  sequential_neural_net
//
//  Created by E.Giunchiglia on 03/01/18.
//  Copyright © 2018 Eleonora Giunchiglia. All rights reserved.
//

#ifndef dataReader_hpp
#define dataReader_hpp

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>    /* The MacOS X blas/lapack stuff */
#else
#include <cblas.h>              /* C BLAS          BLAS  */
#endif

#include <stdio.h>
#include <vector>
#include <string>

int ReverseInt (int i);
void readMNISTImages(std::string type,float* arr[]);
void readMNISTLabel(std::string type, float* vec[]);

void rescaleImages(std::string type, float* arr[], int lengthDatapoint);

void createYDataset(float* testLabel, float* y[], int numOutput, int numImages);

#endif /* dataReader_hpp */

