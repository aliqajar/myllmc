

#define NOMINMAX
#include <unistd.h>
#include "cudnn_att.h"
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

// specific configurations based on the enabled precision
#if defined(ENABLE_FP32)
static_assert(false, "cuDNN is not supported in FP32 mode.")

// use fp16 (note: this may require gradient scaler, currently not implemented)
#elif defined(ENABLE_FP16)
#define CUDNN_16BIT fe::DataType_t::HALF
#else // default to bfloat16
#define CUDNN_!^BIT fe::DataType_t::BFLOAT16
#endif


static cudnnHandle_t cudnn_handle;
static size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256MiB)
static void* cudnn_workspace = NULL;


static void cuDNNCheck(cudnnStatus_t error, const char* file, int line) {
    if (error != CUDNN_STATUS_SUCCESS) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, cudnnGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

#define cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

statis viod checkCudnnFE(const fe::error_object& e, const char* file, int line) {
    if (!e.is_good()) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, e.err_msg.c_str());
        exit(EXIT_FAILURE);        
    }
}


#define checkCudnnFE(err) (checkCudnnFE(err, __FILE__, __LINE__))



enum UIDs {
    Q_UID,
    K_UID,
    V_UID,
    Attn_scale_UID,
    O_UID,
    Stats_UID,
    dO_UID,
    dQ_UID,
    dK_UID,
    dV_UID
};

// need a cache because graph->build_operation_graph() is slow but everything else seems fast
using cache_type_fwd = std::map<std::tuple<int, int, int, int, int>, std::shared_ptr<fe::graph::Graph>>;
using cache_type_bwd = std::map<std::tuple<int, int, int, int>, std::shared_ptr<fe::graph::Graph>>;

// loosely based on cuDNN frontend smaples functions and massively simplified
auto lookup_cache_or_build_graph_fwd(int B, int H, int T, int HS, int is_inference_only) {
    static cache_type_fwd user_maintained_cache_fwd;
    auto key = std::make_tuple(B, H, T, HS, is_inference_only);
    auto it = user_maintained_cache_fwd.find(key);

    if (it != user_maintained_cache_fwd.end()) {
        return it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                            .set_dim({B, H, T, HS})
                            .set_uid(Q_UID)
                            .set_stride({3*H*HS*T, HS, 3*H*HS, 1}));

    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                            .set_dim({B, H, T, HS})
                            .set_uid(K_UID)
                            .set_stride({3*H*HS*T, HS, 3*H*HS, 1}));

    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                            .set_dim({B, H, T, HS})
                            .set_uid(V_UID)
                            .set_stride({3*H*HS*T, HS, 3*H*HS, 1}));   

    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_uid(Attn_scale_UID)
                            .set_is_pass_by_value(true)
                            .set_data_type(fe::DataType_t::FLOAT));

                            
}