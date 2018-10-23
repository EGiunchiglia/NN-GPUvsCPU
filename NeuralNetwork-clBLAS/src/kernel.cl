//
//  kernel.c
//  nn_BLAS_GPU
//
//  Created by E.Giunchiglia on 15/01/18.
//  Copyright © 2018 Eleonora Giunchiglia. All rights reserved.
//

__kernel void tanhNonLinearity(global float *z){
    int i = get_global_id(0);
    z[i] = 1.7159*tanh(2.0/3.0*z[i]);
}

__kernel void expNonLinearity(global float* z){
    int i = get_global_id(0);
    z[i] = exp(z[i]);
}

__kernel void gradientHiddenNonLinearity(global float* result, global float* hiddenNeurons){
    int i = get_global_id(0);
    result[i] = 2.0/3.0 *(1.7159 - pow(hiddenNeurons[i],2));
}

__kernel void elementwiseMultiplication(global float* result, global float* a, global float* b){
    int i = get_global_id(0);
    result[i] = a[i]*b[i];
}

__kernel void elementwiseSubtraction(global float* result, global float* a, global float* b){
    int i = get_global_id(0);
    result[i] = a[i]-b[i];
}

__kernel void subtract(global float* result, global float* a){
    int i = get_global_id(0);
    result[i] -= a[i];
}

__kernel void multiplyConstant(global float* x, global float* alpha){
    int i = get_global_id(0);
    x[i] /= *alpha;
}

