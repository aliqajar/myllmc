

// flash attention


#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H


#include "cuda_common.h"

// forward decalration of functions defined in cudnn_att.cpp

void create_cudnn();
void destroy_cudnn();
void attention_forward_cudnn(floatX* out,   // output: (R, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)   
                             floatX* inp, // input: (B, T, 3, NH, HS) QKV
                             int B, int T, int NH, int C, cudaStream_t stream);


void attention_backward_cudnn(floatX* dqkvr,    // output
                                floatX* dout, floatX* qkvr, floatX* o, float* stats,  // input
                                int B, int T, int NH, int C, cudaStream_t stream);


#endif  // CUDNN_ATT_H