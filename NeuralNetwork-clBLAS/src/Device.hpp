//
//  Device.hpp
//  nn_BLAS_GPU
//
//  Created by E.Giunchiglia on 26/01/18.
//  Copyright © 2018 Eleonora Giunchiglia. All rights reserved.
//

#ifndef Device_hpp
#define Device_hpp

#include <stdio.h>
#include <clBLAS.h>


class Device{

    
private:
    
    cl_mem buf1;
    cl_mem buf2;
    cl_platform_id platform_id;
    cl_uint n_platforms;
    cl_device_id device_id;
    cl_uint n_devices;
    cl_context_properties properties[3];
    cl_int err;
    cl_program program;
    
public:
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel, tanh_kernel, exp_kernel, subtract_kernel, nonLinearity_kernel, mul_kernel, multiplyConstant_kernel, elementwiseSubtraction_kernel;
    size_t group_pattern[];
public:
    
    Device(char* source_str);
    ~Device();
};

#endif /* Device_hpp */
