#ifdef GGML_USE_CUBLAS
#include "ggml/src/ggml-cuda.h"
#elif defined(GGML_USE_CLBLAST)
#include "ggml/src/ggml-opencl.h"
#endif

bool rwkv_gpu_offload_layers(struct rwkv_context * ctx, const uint32_t n_layers) {
#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
    const auto offload = [&](struct ggml_tensor * tensor) {
        // TODO Support multi-GPU
        tensor->backend = GGML_BACKEND_GPU;
#ifdef GGML_USE_CUBLAS
        ggml_cuda_transform_tensor(tensor->data, tensor);
#elif defined(GGML_USE_CLBLAST)
        ggml_cl_transform_tensor(tensor->data, tensor);
#endif
    };

    const size_t n_gpu = std::min(n_layers, ctx->model->header.n_layer);

    if (ctx->model->offloaded_layer_count < n_gpu) {
        for (size_t & i = ctx->model->offloaded_layer_count; i < n_gpu; i++) {
            const struct rwkv_layer & layer = ctx->model->layers[i];

            // TODO Also offload other operations to GPU with ggml_cuda_assign_buffers
            offload(layer.att_key);
            offload(layer.att_value);
            offload(layer.att_receptance);
            offload(layer.att_output);

            offload(layer.ffn_key);
            offload(layer.ffn_value);
            offload(layer.ffn_receptance);
        }

        return true;
    }
#endif
    return false;
}
