// Tests that quantized matmul on GPU works.
#include <stdlib.h>
#include <stdio.h>

#if defined(GGML_USE_CUDA) || defined(GGML_USE_METAL)

#include <math.h>
#include <string.h>

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#if defined(GGML_USE_CUDA)
#include "ggml/include/ggml-cuda.h"
#elif defined(GGML_USE_METAL)
#include "ggml/include/ggml-metal.h"
#endif

#include "assertions.inc"

// ELEMENT_COUNT >= 64 makes metal kernel happy
#define ELEMENT_COUNT 64

int main(void) {
    struct ggml_init_params params = {
        .mem_size   = 96 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };

#ifdef GGML_USE_CUDA
    ggml_backend_t backend = ggml_backend_cuda_init(0);
#elif defined(GGML_USE_METAL)
    ggml_backend_t backend = ggml_backend_metal_init();
#endif

    ggml_backend_t backend_cpu = ggml_backend_cpu_init();

    ggml_backend_t backends[2] = { backend, backend_cpu };

    ASSERT(backend && backend_cpu, "ggml_backend init failed\n");

    struct ggml_context * ctx = ggml_init(params);

    // ---

    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ELEMENT_COUNT, 1);
    struct ggml_tensor * x_quantized = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_0, ELEMENT_COUNT, 1);

    struct ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ELEMENT_COUNT);

    struct ggml_tensor * mul0 = ggml_mul_mat(ctx, x, y);
    struct ggml_tensor * mul1 = ggml_mul_mat(ctx, x_quantized, y);

    ggml_backend_buffer_t buffer_gpu = ggml_backend_alloc_buffer(backend, ggml_nbytes(x_quantized) + 1024);
    ggml_backend_buffer_t buffer_cpu = ggml_backend_alloc_buffer(backend_cpu, ggml_nbytes(x) + ggml_nbytes(y) + 1024);

    ggml_backend_buffer_set_usage(buffer_gpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(buffer_cpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    struct ggml_tallocr tallocr_gpu = ggml_tallocr_new(buffer_gpu);
    struct ggml_tallocr tallocr_cpu = ggml_tallocr_new(buffer_cpu);

    ggml_tallocr_alloc(&tallocr_gpu, x_quantized);
    ggml_tallocr_alloc(&tallocr_gpu, mul1);
    ggml_tallocr_alloc(&tallocr_cpu, x);
    ggml_tallocr_alloc(&tallocr_cpu, y);
    ggml_tallocr_alloc(&tallocr_cpu, mul0);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, mul0);
    ggml_build_forward_expand(graph, mul1);

    // ---

    float * data = (float *) malloc(ELEMENT_COUNT * ggml_type_size(GGML_TYPE_F32));
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        if (i % 2 == 0)
            data[i] = 1.0F * i / 2;
        else
            data[i] = 0;
    }
    uint8_t * data_quantized = (uint8_t *) malloc(ELEMENT_COUNT * ggml_type_size(GGML_TYPE_Q5_0));
    ggml_quantize_chunk(x_quantized->type, (const float *) data, data_quantized, 0, 1, ELEMENT_COUNT, NULL);

    memcpy(x->data, data, ggml_nbytes(x));
    memcpy(y->data, data, ggml_nbytes(y));

    // ---

#if defined(GGML_USE_METAL)
    memcpy(x_quantized->data, data_quantized, ggml_nbytes(x_quantized));
#else
    ggml_backend_tensor_set(x_quantized, data_quantized, 0, ggml_nbytes(x_quantized));
#endif

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 2, 4096, false);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_graph_compute(sched, graph);

    float result0;
    float result1;

    ggml_backend_tensor_get(mul0, &result0, 0, ggml_nbytes(mul0));
    ggml_backend_tensor_get(mul1, &result1, 0, ggml_nbytes(mul1));

    fprintf(stderr, "FP32 CPU result = %f\n", (double)result0);
    fprintf(stderr, "Q5_0 GPU result = %f\n", (double)result1);

    ASSERT(fabsf(result0 - result1) <= 100.0F, "Results differ too much");

    ggml_free(ctx);
    free(data);
    free(data_quantized);

    return 0;
}

#else

int main(void) {
    fprintf(stderr, "Skipping test_quantized_matmul_on_gpu.c: GGML_USE_CUDA is not defined\n");

    return 0;
}

#endif
