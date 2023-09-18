// Tests that quantized matmul on GPU works.
#include <ggml.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// TODO Move to inc
#define ASSERT(x, ...) {\
        if (!(x)) {\
            fprintf(stderr, "*** Assertion failed ***\n");\
            fprintf(stderr, __VA_ARGS__);\
            fprintf(stderr, "\n%s:%d\n", __FILE__, __LINE__);\
            abort();\
        }\
    }

#define SET_ELEMENT_F32(tensor, i, value) ((float *) tensor->data)[i] = value

#define ELEMENT_COUNT 32

int main(void) {
    #ifdef GGML_USE_CUBLAS

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // ---

    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ELEMENT_COUNT, 1);

    for (int i = 0; i < ELEMENT_COUNT; i++) {
        SET_ELEMENT_F32(x, i, 1.0F * i);
    }

    // ---

    struct ggml_tensor * x_quantized = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, ELEMENT_COUNT, 1);

    int64_t hist[16];
    ggml_quantize_chunk(x_quantized->type, (const float *) x->data, x_quantized->data, 0, ELEMENT_COUNT, hist);

    x_quantized->backend = GGML_BACKEND_GPU;
    ggml_cuda_transform_tensor(x_quantized->data, x_quantized);

    // ---

    struct ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ELEMENT_COUNT);

    for (int i = 0; i < ELEMENT_COUNT; i++) {
        SET_ELEMENT_F32(y, i, 1.0F * i);
    }

    // ---

    struct ggml_tensor * mul0 = ggml_mul_mat(ctx, x, y);
    struct ggml_tensor * mul1 = ggml_mul_mat(ctx, x_quantized, y);

    struct ggml_cgraph graph = ggml_build_forward(mul0);

    ggml_build_forward_expand(&graph, mul1);

    struct ggml_cplan plan = ggml_graph_plan(&graph, 2);

    uint8_t * work_data = (uint8_t *) malloc(plan.work_size);
    plan.work_data = work_data;

    ggml_graph_compute(&graph, &plan);

    free(work_data);

    float result0 = ((float *) mul0->data)[0];
    float result1 = ((float *) mul1->data)[0];

    fprintf(stderr, "FP32 CPU result = %f\n", result0);
    fprintf(stderr, "Q4_0 GPU result = %f\n", result1);

    ASSERT(fabsf(result0 - result1) <= 100.0F, "Results differ too much");

    ggml_free(ctx);

    #endif

    return 0;
}
