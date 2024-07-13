#if defined(GGML_USE_CUDA) || defined(GGML_USE_CLBLAST)

#if defined(GGML_USE_CUDA)
#    include "ggml/include/ggml-cuda.h"
#elif defined(GGML_USE_CLBLAST)
#    include "ggml/src/ggml-opencl.h"
#endif

// API function.
bool rwkv_gpu_offload_layers(struct rwkv_context * ctx, const uint32_t n_layers) {
    const auto offload = [&](struct ggml_tensor * tensor) {
        // TODO Support multi-GPU
        tensor->backend = GGML_BACKEND_TYPE_GPU;
// #if defined(GGML_USE_CUDA)
//         ggml_cuda_transform_tensor(tensor->data, tensor);
// #elif defined(GGML_USE_CLBLAST)
//         ggml_cl_transform_tensor(tensor->data, tensor);
// #endif
    };

    const size_t n_gpu = std::min(n_layers, ctx->model->header.n_layer + 1);

    if (ctx->model->offloaded_layer_count >= n_gpu) {
        return false;
    }

    for (size_t & i = ctx->model->offloaded_layer_count; i < n_gpu; i++) {
        if (i == ctx->model->header.n_layer) {
            // This is the index of the model head.
            offload(ctx->model->head);

            continue;
        }

        const struct rwkv_layer & layer = ctx->model->layers[i];

        // TODO Also offload other supported operations to GPU
        offload(layer.att_key);
        offload(layer.att_value);
        offload(layer.att_receptance);
        offload(layer.att_output);

        if (layer.att_gate != NULL) {
            offload(layer.att_gate);
        }

        offload(layer.ffn_key);
        offload(layer.ffn_value);
        offload(layer.ffn_receptance);
    }

    return true;
}

#else

// API function.
bool rwkv_gpu_offload_layers(struct rwkv_context * ctx, const uint32_t n_layers) {
    return false;
}

#endif
