// Tests that quantized matmul on GPU works.
#include <stdlib.h>
#include <stdio.h>

#if defined(GGML_USE_CUBLAS)

#include <math.h>

#include <ggml.h>
#include "ggml/include/ggml-cuda.h"

#include "assertions.inc"

#define SET_ELEMENT_F32(tensor, i, value) ((float *) tensor->data)[i] = value

#define ELEMENT_COUNT 32

int main(void) {
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

    ggml_quantize_chunk(x_quantized->type, (const float *) x->data, x_quantized->data, 0, 1, ELEMENT_COUNT, NULL);

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

    // Allocation on heap instead of stack avoids SegFault when GGML_MAX_NODES is set to a large value.
    struct ggml_cgraph * graph = (struct ggml_cgraph *) calloc(1, sizeof(struct ggml_cgraph));
    ggml_build_forward_expand(graph, mul0);
    ggml_build_forward_expand(graph, mul1);

    struct ggml_cplan * plan = ggml_graph_plan(graph, 2);

    uint8_t * work_data = (uint8_t *) malloc(plan->work_size);
    plan->work_data = work_data;

    ggml_graph_compute(graph, plan);

    free(plan);
    free(graph);
    free(work_data);

    float result0 = ((float *) mul0->data)[0];
    float result1 = ((float *) mul1->data)[0];

    fprintf(stderr, "FP32 CPU result = %f\n", result0);
    fprintf(stderr, "Q4_0 GPU result = %f\n", result1);

    ASSERT(fabsf(result0 - result1) <= 100.0F, "Results differ too much");

    ggml_free(ctx);

    return 0;
}

#else

int main(void) {
    fprintf(stderr, "Skipping test_quantized_matmul_on_gpu.c: GGML_USE_CUBLAS is not defined\n");

    return 0;
}

#endif
