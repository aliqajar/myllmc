

// GPT-2 transformer neural net training loop. 

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>

// --------- CPU utilities ----------------------
#include "llmc/utils.h"
#include "llmc/tokenizer.h"
#include "llmc/dataloader.h"
#include "llmc/rand.h"
#include "llmc/schedulers.h"
#include "llmc/sampler.h"
#include "llmc/logger.h"
#include "llmc/mfu.h"
#include "llmc/outlier_detector.h"

// --------- GPU Utilities ---------------------
#include "llmc/cuda_common.h"
#include "llmc/cuda_utils.cuh"
#include "llmc/cublas_common.h"
#include "llmc/encoder.cuh"
#include "llmc/layernorm.cuh"
#include "llmc/matmul.cuh"

#ifdef ENABLE_CUDNN
#include "llmc/cudnn_att.h"
#else
#include "llmc/attention.cuh"
#endif

#include "llmc/adamw.cuh"
#include "llmc/global_norm.cuh"
#include "llmc/zero.cuh"


// ---------------------------------------------
// global vars for I/O
char filename_buffer[512];

// global vars containing information about the GPU this process is running on
constexpr const size_t IO_BUF_SIZE = 32 * 1024 * 1024;

// ---------------------------------------------
// GPT-2 model definition


typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int num_layers;
    int num_heads;
    int channels;
} GPT2Config;


// the parameters of the model
constexpr const int NUM_PARAMETER_TENSORS = 16;
typedef struct {
    floatX* wte;
    floatX* wpe;
    floatX* ln1w;
    floatX* ln1b;
    floatX* qkvw;
    floatX* qkvb;
    floatX* attprojw;
    floatX* attprojb;
    floatX* ln2w;
    floatX* ln2b;
    floatX* fcw;
    floatX* fcb;
    floatX* fcprojw;
    floatX* fcprojb;
    floatX* lnfw;
    floatX* lnfb;
} ParameterTensors;

static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");


void fill_in_paramter_sizes(size_t* param_sizes, size_t* param_sizeof, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C =  config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;

    param_sizes[0] = Vp * C;
    param_sizes[1] = maxT * C;
    param_sizes[2] = L * C;
    param_sizes[3] = L * C;
    param_sizes[4] = L * (3 * C) * C;
    param_sizes[5] = L * (3 * C);
    param_sizes[6] = L * C * C;
    param_sizes[7] = L * C;
    param_sizes[8] = L * C;
    param_sizes[9] = L * C;
    param_sizes[10] = L * (4 * C) * C;
    param_sizes[11] = L * (4 * C);
    param_sizes[12] = L * C * (4 * C);
    param_sizes[13] = L * C;
    param_sizes[14] = C;
    param_sizes[15] = C;

    // populate the parameter sizes in bytes (all the same for noew, keeping for future use)
    for (int i=0; i < NUM_PARAMETER_TENSORS; i++) {
        param_sizeof[i] = sizeof(floatX)
    }
}


// allocate memory for the parameters and poitn the individual tensors to the right places
void* malloc_and_point_parameters(ParameterTensors* params, size_t* param_elements, size_t* param_sizeof) {
    // calculate the total numerb of parameters and bytes across all tensors
    size_t num_parameters_bytes = 0;
    for (int i = 0; i< NUM_PARAMETER_TENSORS; i++) {
        num_parameters_bytes += param_elements[i] * param_sizeof[i];
    }

    // malloc all parameters all at once on the device
    void* params_memory;

    cudaCheck(cudaMalloc((void**)&params_memory, num_parameters_bytes));

    // assign all the tensors their place in the array
    floatX** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };

    char* params_memory_iterator = (char*)params_memory;
    for(int i=0; i< NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX*)params_memory_iterator;
        params_memory_iterator += param_elements[i] * param_sizeof[i];
    }

    return params_memory;
}


constexpr int NUM_ACTIVATION_TENSORS = 21;
typedef struct {
    floatX* encoded;  // (B, T, C)
    floatX* ln1;    // (L, B, T C)
    floatX* ln1_mean;  // (L, B, T)
    floatX* ln1_rstd;   // (L, B, T)
    floatX* atty; // (L, B, T C)
    
    // cuDNN saves only some statistics info

#if ENABLE_CUDNN
    float* att; // (L, B, NH, T)
#else
    floatX* att; // (L, B, NH, T, T)
#endif

    floatX* residual2; // (L, B, T, C)
    floatX* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    floatX* fch; // (L, B, T, 4*C)
    floatX* fch_gelu; // (L, B, T, 4*C)
    floatX* residual3; // (L, B, T, C)
    floatX* lnf; // (B, T, C);   if LN recomputation is enabled (-r 2 and above), will be used for _all_ layernorms
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* losses; // (B, T), will be accumulated in micro-steps
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    floatX* qkvr; // (L, B, T, 3*C)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    floatX* output;

    // some additional scratch buffers
    floatX* scratch_bt4c;   // (B, T, 4*C)
    floatX* scratch_btc;    // (B, T, C)
} ActivationTensors;


struct TensorSpec {
    void** ptr;
    size_t size;
    DType type;
};