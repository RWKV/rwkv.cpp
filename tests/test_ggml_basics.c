// Tests that ggml basics work.

#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SET_ELEMENT_F32(tensor, i, value) ((float *) tensor->data)[i] = value

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
        ASSERT(fabsf(actual - expected_value) <= 0.0000001F, "At %s[%d]: expected %f, actual %f", #tensor, i, expected_value, actual);\
    }

int main(int argc, const char ** argv) {
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
    graph.n_threads = 2;
    ggml_graph_compute(ctx, &graph);

    ASSERT_ELEMENT_F32(sum, 0, -9.0F);
    ASSERT_ELEMENT_F32(sum, 1, 2.0F);
    ASSERT_ELEMENT_F32(sum, 2, 5.5F);
    ASSERT_ELEMENT_F32(sum, 3, 9.0F);

    ggml_print_objects(ctx);

    ggml_free(ctx);

    return 0;
}
