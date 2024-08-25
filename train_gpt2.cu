

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

}