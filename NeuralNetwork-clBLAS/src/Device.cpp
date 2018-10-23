//
//  Device.cpp
//  nn_BLAS_GPU
//
//  Created by E.Giunchiglia on 26/01/18.
//  Copyright © 2018 Eleonora Giunchiglia. All rights reserved.
//

#include "Device.hpp"


Device:: Device(char* source_str){
    platform_id=0;
    device_id=0;
    queue=0;
    
    //Get 1 platform
    if(clGetPlatformIDs(1, &platform_id, &n_platforms) != CL_SUCCESS){
        cerr << "error: no platform" << endl;
        exit(1);
    };
    
    //On that platform get 1 device of desired kind
    if(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices) != CL_SUCCESS){
        cerr << "error: no device" << endl;
        exit(1);
    };
    
    //Set the properties list. The properties list must end with 0.
    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = (cl_context_properties) platform_id;
    properties[2] = 0;
    
    //Create a context for device.
    context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);
    if(err != CL_SUCCESS){
        cerr << "error in creating context" << endl;
        exit(1);
    };
    
    
    //Create a queue of commands to be sent from host to device.
    queue = clCreateCommandQueueWithProperties(context , device_id, 0, &err);
    if(err != CL_SUCCESS){
        cerr << "error in creating the command queue" << endl;
        clReleaseContext(context);
        exit(1);
    };
    
    group_pattern[0] = size_t(15);
    group_pattern[1] = size_t(15);
    group_pattern[2] = 0;
    
    /////////
    // CREATE KERNEL PROGRAMS
    ////////
    //Build the kernel program for tanh
    program = clCreateProgramWithSource(context, 1, (const char**) &source_str, NULL, &err);
    if(err != CL_SUCCESS){
        cerr << "error in creating kernel program for tanh non-linearity. Error: " << err << endl;
        exit(1);
    }
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err!=CL_SUCCESS){
        cerr << "error in building kernel program. Error: " << err << endl;
        exit(1);
    }
    tanh_kernel = clCreateKernel(program, "tanhNonLinearity", &err);
    if(err != CL_SUCCESS){
        cerr << "error in creating kernel executable for tanh non-linearity. Error: " << err << endl;
        exit(1);
    }
    exp_kernel = clCreateKernel(program, "expNonLinearity", &err);
    if(err != CL_SUCCESS){
        cerr << "error in creating kernel executable for exp. Error: " << err << endl;
        exit(1);
    }
    //Build the kernel program for subtract
    subtract_kernel = clCreateKernel(program, "subtract", &err);
    if(err != CL_SUCCESS){
        cerr << "error in creating kernel executable for subtract. Error: " << err << endl;
        exit(1);
    }
    //Build the kernel program for gradient hidden input non linearity
    nonLinearity_kernel = clCreateKernel(program, "gradientHiddenNonLinearity", &err);
    if(err != CL_SUCCESS){
        cerr << "error in creating kernel executable for get gradient hidden non linearity. Error: " << err << endl;
        exit(1);
    }
    //Build the kernel program for elementwiseMultipication
    mul_kernel = clCreateKernel(program, "elementwiseMultiplication", &err);
    if(err != CL_SUCCESS){
        cerr << "error in creating kernel executable for multiplication kernel. Error: " << err << endl;
        exit(1);
    }
    //Build the kernel program for constant multiplication
    multiplyConstant_kernel = clCreateKernel(program, "multiplyConstant", &err);
    if(err != CL_SUCCESS){
        cerr << "error in creating kernel executable for constant multiplication. Error: " << err << endl;
        exit(1);
    }
    
    //Build the kernel program for constant multiplication
    elementwiseSubtraction_kernel = clCreateKernel(program, "elementwiseSubtraction", &err);
    if(err != CL_SUCCESS){
        cerr << "error in creating kernel executable for elementiwise subtraction. Error: " << err << endl;
        exit(1);
    }
    
    
    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(1);
    }
}

Device::~Device(){
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    // Finalize work with clblas.
    clblasTeardown();
}
