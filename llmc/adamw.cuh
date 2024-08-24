

#include "cuda_common.h"
#include "cuda_utils.h"

// cuda kernels

// implements linear interpolation using two floating-point operations (as opposed to three in a naive implementation)
__device__ float lerp(float start, float end, float weight) {
    return fma(weight, enf, fma(-weight, start, start))
}

template <typename Tp, typename Tg>
__device__ void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters, 
    float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
    float grad_scale, unsigned int seed) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_parameters) {return;} // guard

        // get the gradient, m, and v for this paramter
        float grad = grad_scale * (float)grads_memory[idx];
        float m = m_memory[idx];
        float v = v_memory[idx];

        // update the first moment (momentum)
        m = lerp(grad, m, beta1)
        m_memory[idx] = m;
        // update the second moment (RMSprop)
        v = lerp(grad * grad, v, beta2);
        v_memory[idx] = v;
        m /= beta1_correction; // m_hat
        v /= beta2_correction; // v_hat

        float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float) params_memory[idx];

        // update this parameter
        float param = old_param - (learning_rate * (m/(sqrtf(v) + eps) + weight_decay * old_param));

        // update our low precision version of the paramters using stochastic rounding
        // this will be used in the next forward pass
        stochastic_rounding(param, &params_memory[idx], seed);
        // write the full, float version of the param into our master copy, if we maintain one
        // this will be used in the next update
        if (master_params_memory != NULL) { master_params_memory[idx] = params; }

    }


template <typename Tp, typename Tg>
__global__ void adamw_kernel3(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_paramters,
                ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride, 
                float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                float grad_scale, unsinged int seed) {
                    adamw_update(params_memory + blockIdx.y * w_stride,
                                master_params_memory ? master_params_memory + blockIdx.y * s_stride : NULL,
                                grads_memory + blockIdx.y * g_stride,
                                m_memory + blockIdx.y * s_stride,
                                v_memory + blockIdx.y * s_stride,
                                num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale, 
                                seed);

                }


template <typename Tp>
__global__ void init_from_master_kernel(Tp* params_memory, float* master_params_memory, size_t num_paramaters,
            ptrdiff_t w_stride, ptrdiff_t s_stride, unsigned int seed) {

                size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= num_paramaters) {return ;}

                params_memory += blockIdx.y * w_stride; // adjust for layer offset
                master_params_memory += blockIdx.y * s_stride;
                stochastic_rounding(master_params_memory[idx], &params_memory[idx], seed);

            }

template <typename Tp, typename Tg>
void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                    ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride, int num_slices, float leaarning_rate, float beta1, float beta2, int t, float eps, float weight_decay, 
                    flaot grad_scale, unsigned int, seed, cudastream_t stream) {
                        // adamw update
                        int block_size = 512;
                        int num_blocks = CEIL_DIV(num_paramters, block_size);
                        float beta1_correction = 1.0f - powf(beta1, t);
                        float beta2_correction = 1.0f - powf(beta2, t);
                        adamw_kernel3<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>(params_memory, master_params_memory, grads_memory,
                                            m_memory, v_memory, num_parameters, w_stride, g_stirde, s_stride,
                                            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale, seed);

                        cudaCheck(cudaGetLastError());
                    }



template <typename TP>
void init_from_master(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                ptrdiff_t w_stirde, ptrdiff_t s_stride, int num_slices, unsigned int seed, cudastream_t stream) {
                    int block_size = 512; // must match block size of adamw_update so that RNG also matches
                    int num_blocks = CEIL_DIV(num_paramaters, block_size);
                    init_from_master_kernel<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>
                    (params_memory, master_params_memory, num_parameters, w_stride, s_stride, seed);

                    cudaCheck(cudaGetLastError());
                }