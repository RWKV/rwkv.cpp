// Tests that ggml basics work.

// Fix build on Linux.
// https://stackoverflow.com/questions/8518264/where-is-the-declaration-of-cpu-alloc
#if defined(__linux__)
#define _GNU_SOURCE
#include <sched.h>
#endif

#include <ggml.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SET_ELEMENT_F32(tensor, i, value) ((float *) tensor->data)[i] = value

// TODO Move to inc
#define ASSERT(x, ...) {\
        if (!(x)) {\
            fprintf(stderr, "*** Assertion failed ***\n");\
            fprintf(stderr, __VA_ARGS__);\
            fprintf(stderr, "\n%s:%d\n", __FILE__, __LINE__);\
            abort();\
        }\
    }

#define ASSERT_ELEMENT_F32(tensor, i, expected_value) {\
        float actual = ((float *) tensor->data)[i];\
        ASSERT(fabsf(actual - expected_value) <= 0.0000001F, "At %s[%d]: expected %f, actual %f", #tensor, i, (double) expected_value, (double) actual);\
    }

// Tests simple computation in a single context.
static void test_computation(void) {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    SET_ELEMENT_F32(x, 0, -10.0F);
    SET_ELEMENT_F32(x, 1, 0.0F);
    SET_ELEMENT_F32(x, 2, 2.5F);
    SET_ELEMENT_F32(x, 3, 5.0F);

    struct ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    SET_ELEMENT_F32(y, 0, 1.0F);
    SET_ELEMENT_F32(y, 1, 2.0F);
    SET_ELEMENT_F32(y, 2, 3.0F);
    SET_ELEMENT_F32(y, 3, 4.0F);

    struct ggml_tensor * sum = ggml_add(ctx, x, y);

    struct ggml_cgraph graph = ggml_build_forward(sum);
    struct ggml_cplan plan = ggml_graph_plan(&graph, 2);
    ggml_graph_compute(&graph, &plan);

    ASSERT_ELEMENT_F32(sum, 0, -9.0F);
    ASSERT_ELEMENT_F32(sum, 1, 2.0F);
    ASSERT_ELEMENT_F32(sum, 2, 5.5F);
    ASSERT_ELEMENT_F32(sum, 3, 9.0F);

    ggml_print_objects(ctx);

    ggml_free(ctx);
}

// Tests that operations on tensors from different contexts work.
// RWKV model loading code depends on this behavior.
static void test_tensors_from_different_contexts(void) {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_context * ctx1 = ggml_init(params);
    struct ggml_context * ctx2 = ggml_init(params);

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 4);
    SET_ELEMENT_F32(x, 0, -10.0F);
    SET_ELEMENT_F32(x, 1, 0.0F);

    struct ggml_tensor * y = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, 4);
    SET_ELEMENT_F32(y, 0, 1.0F);
    SET_ELEMENT_F32(y, 1, 2.0F);

    struct ggml_tensor * sum = ggml_add(ctx2, x, y);

    struct ggml_cgraph graph = ggml_build_forward(sum);
    struct ggml_cplan plan = ggml_graph_plan(&graph, 2);
    ggml_graph_compute(&graph, &plan);

    ASSERT_ELEMENT_F32(sum, 0, -9.0F);
    ASSERT_ELEMENT_F32(sum, 1, 2.0F);

    ggml_free(ctx0);
    ggml_free(ctx1);
    ggml_free(ctx2);
}

int main(void) {
    test_computation();

    test_tensors_from_different_contexts();

    return 0;
}
