//
//  dataReader.cpp
//  sequential_neural_net
//
//  Created by E.Giunchiglia on 03/01/18.
//  Copyright © 2018 Eleonora Giunchiglia. All rights reserved.
//
//
//  The following code has been taken and modified from:
//  http://eric-yuan.me/cpp-read-mnist/
//


#include "dataReader.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>



using namespace std;

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
void readMNISTImages(string type, cl_float* arr[])
{
    int numberOfImages=0;
    int DataOfAnImage=784;
    string path;
    
    if(type.compare("train") == 0){
        numberOfImages = 60000;
        path = "./data/train-images.idx3-ubyte";
    }
    else if(type.compare("test") == 0){
        numberOfImages = 10000;
        path = "./data/t10k-images.idx3-ubyte";
    }
    else{
        cerr << "Given type of dataset is wrong" << endl;
        exit(1);
    }
    
    *arr = new cl_float[numberOfImages*DataOfAnImage];
    
    ifstream file;
    file.exceptions (ifstream::failbit | ifstream::badbit);
    
    try{
        file.open(path,ios::binary);
        if (file.is_open()) {
            int magic_number=0;
            int number_of_images=0;
            int n_rows=0;
            int n_cols=0;
            
            file.read((char*)&magic_number,sizeof(magic_number));
            magic_number= ReverseInt(magic_number);
            file.read((char*)&number_of_images,sizeof(number_of_images));
            number_of_images= ReverseInt(number_of_images);
            file.read((char*)&n_rows,sizeof(n_rows));
            n_rows= ReverseInt(n_rows);
            file.read((char*)&n_cols,sizeof(n_cols));
            n_cols= ReverseInt(n_cols);
            
            
            for(int i=0;i<number_of_images;++i){
                for(int r=0;r<n_rows;++r){
                    for(int c=0;c<n_cols;++c){
                        unsigned char temp=0;
                        file.read((char*)&temp,sizeof(temp));
                        (*arr)[(i*DataOfAnImage)+ (n_rows*r)+c]= (cl_float)temp;
                    }
                }
            }
        }
    } catch(ifstream::failure e){
        cerr << "Exception raised while handling the file" << endl;
        exit(1) ;
    }
}

void readMNISTLabel(string type, cl_float* vec[])
{
    int numberOfImages;
    string path;
    
    if(type.compare("train") == 0){
        path = "./data/train-labels.idx1-ubyte";
        numberOfImages = 60000;
    }
    else if(type.compare("test") == 0){
        path = "./data/t10k-labels.idx1-ubyte";
        numberOfImages = 10000;
    }
    else{
        cerr << "Given type of dataset is wrong" << endl;
        exit(1);
    }
    
    ifstream file;
    file.exceptions (ifstream::failbit | ifstream::badbit);
    
    try{
        file.open(path, ios::binary);
        if (file.is_open())
        {
            *vec = new cl_float[numberOfImages];
            
            int magic_number = 0;
            int number_of_images = 0;
            file.read((char*) &magic_number, sizeof(magic_number));
            magic_number = ReverseInt(magic_number);
            file.read((char*) &number_of_images,sizeof(number_of_images));
            number_of_images = ReverseInt(number_of_images);
            for(int i = 0; i < number_of_images; ++i){
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                (*vec)[i]= (cl_float)temp;
            }
        }
    } catch(ifstream::failure e){
        cerr << "Exception raised while handling the file" << endl;
        exit(1) ;
    }
}

//Since we are taking as input black and white images in which each pixel has value in range [0, 255]
//we have to rescale their values in the range [0, 1] between inputing them in the neural network.
void rescaleImages(string type, cl_float* images[], int lengthDatapoint){
    int numberOfImages = 0;
    if(type.compare("train") == 0){
        numberOfImages = 60000;
    }
    else if(type.compare("test") == 0){
        numberOfImages = 10000;
    }
    else{
        cerr << "Given type of dataset is wrong" << endl;
        exit(1);
    }
    for (int i=0; i<numberOfImages; i++){
        for (int j=0; j < lengthDatapoint; j++){
            (*images)[i*lengthDatapoint+j] /= cl_float(255.0);
        }
    }
}



void createYDataset(cl_float *labels, cl_float* y[], int numOutput, int numImages){
    *y = new cl_float[numImages*numOutput];
    for(int i=0; i<numImages; i++){
        cl_float temp = labels[i];
        for(int j=0; j<numOutput;j++){
            if(j!=temp)
                (*y)[i*numOutput+j] = 0;
            else
                (*y)[i*numOutput+j] = 1;
        }
    }
}


