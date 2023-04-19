// Tests that Q4_1_O matmul on a large matrix works (does not crash, etc.)

#include "ggml.h"
#include "rwkv.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GGML_GET_ELEMENT_F32(tensor, i) (((float *) tensor->data)[i])

#define GGML_SET_ELEMENT_F32(tensor, i, value) ((float *) tensor->data)[i] = value

#define GGML_ASSERT(x, ...) {\
        if (!(x)) {\
            fprintf(stderr, "*** Assertion failed ***\n");\
            fprintf(stderr, __VA_ARGS__);\
            fprintf(stderr, "\n%s:%d\n", __FILE__, __LINE__);\
            abort();\
        }\
    }

#define RANDOM_FLOAT() (((rand() & 0xFFF) / ((float) 0xFFF) - 0.5F) * 3.0F)

// ---

#define QK 32
#define MATRIX_SIZE 1024

int main(int argc, const char ** argv) {
    srand(42);

    struct ggml_init_params params = {
        .mem_size   = 8 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    fprintf(stderr, "%d\n", __LINE__);

    struct ggml_context * ctx = ggml_init(params);

    fprintf(stderr, "%d\n", __LINE__);

    struct ggml_tensor * mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MATRIX_SIZE, MATRIX_SIZE);

    fprintf(stderr, "%d\n", __LINE__);

    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        GGML_SET_ELEMENT_F32(mat, i, RANDOM_FLOAT());
    }

    fprintf(stderr, "%d\n", __LINE__);

    // Add some outliers
    for (int i = 0; i < MATRIX_SIZE; i++) {
        GGML_SET_ELEMENT_F32(mat, i * MATRIX_SIZE + 1, RANDOM_FLOAT() * 100.0F);
    }

    fprintf(stderr, "%d\n", __LINE__);

    struct ggml_tensor * quantized_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1_O, MATRIX_SIZE, MATRIX_SIZE);

    fprintf(stderr, "%d\n", __LINE__);

    int64_t histogram[16];

    fprintf(stderr, "%d\n", __LINE__);

    ggml_quantize_q4_1_o(mat->data, quantized_mat->data, MATRIX_SIZE * MATRIX_SIZE, QK, histogram);

    fprintf(stderr, "%d\n", __LINE__);

    struct ggml_tensor * vec = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, MATRIX_SIZE);

    fprintf(stderr, "%d\n", __LINE__);

    for (int i = 0; i < MATRIX_SIZE; i++) {
        GGML_SET_ELEMENT_F32(vec, i, RANDOM_FLOAT());
    }

    fprintf(stderr, "%d\n", __LINE__);

    struct ggml_tensor * expected_result = ggml_mul_mat(ctx, mat, vec);

    fprintf(stderr, "%d\n", __LINE__);
    struct ggml_tensor * quantized_result = ggml_mul_mat(ctx, quantized_mat, vec);

    fprintf(stderr, "%d\n", __LINE__);

    struct ggml_cgraph graph = ggml_build_forward(expected_result);

    fprintf(stderr, "%d\n", __LINE__);
    ggml_build_forward_expand(&graph, quantized_result);

    fprintf(stderr, "%d\n", __LINE__);
    graph.n_threads = 4;

    fprintf(stderr, "%d\n", __LINE__);
    ggml_graph_compute(ctx, &graph);

    fprintf(stderr, "%d\n", __LINE__);

    float diff_sum = 0.0F;

    for (int i = 0; i < MATRIX_SIZE; i++) {
        diff_sum += fabsf(GGML_GET_ELEMENT_F32(expected_result, i) - GGML_GET_ELEMENT_F32(quantized_result, i));
    }

    fprintf(stderr, "%d\n", __LINE__);

    float diff_average = diff_sum / MATRIX_SIZE;

    fprintf(stderr, "%d\n", __LINE__);

    // More strict test is in test_Q4_1_O.c, here we just do sanity check
    GGML_ASSERT(diff_average <= 2.0F, "Unexpected average difference value %f", diff_average);

    fprintf(stderr, "%d\n", __LINE__);

    ggml_free(ctx);

    fprintf(stderr, "%d\n", __LINE__);

    return 0;
}
