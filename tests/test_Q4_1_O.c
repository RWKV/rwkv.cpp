// Tests that Q4_1_O basics (quantization, dequantization, matmul) work.

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

#define GGML_ASSERT_ELEMENT_F32(tensor, i, expected_value) {\
        float actual = GGML_GET_ELEMENT_F32(tensor, i);\
        GGML_ASSERT(fabsf(actual - expected_value) <= 0.0000001F, "At %s[%d]: expected %f, actual %f", #tensor, i, expected_value, actual);\
    }

#define RANDOM_FLOAT() (((rand() & 0xFFF) / ((float) 0xFFF) - 0.5F) * 3.0F)

// ---

#define QK4_1_O 32

// Copied from ggml.c
typedef struct {
    ggml_fp16_t d;
    ggml_fp16_t m;
    uint16_t outlier_index;
    ggml_fp16_t outlier_value;
    uint8_t qs[QK4_1_O / 2];
} block_q4_1_o;

static_assert(sizeof(block_q4_1_o) == 8 + QK4_1_O / 2, "Wrong q4_1_o block size/padding");

int main(int argc, const char ** argv) {
    // Needed to initialize FP16 lookup table
    {
        struct ggml_init_params params = { 0, NULL };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    fprintf(stderr, "System info: %s\n", rwkv_get_system_info_string());

    quantize_fns_t quantize_fns = ggml_internal_get_quantize_fn(GGML_TYPE_Q4_1_O);

    float src[QK4_1_O];
    uint8_t dest[24];

    // 1..32
    for (int i = 0; i < QK4_1_O; i++) {
        src[i] = (float) (i + 1);
    }

    // --- Quantization ---
    (quantize_fns.quantize_row_q)(src, dest, QK4_1_O);

    float delta_result = ggml_fp16_to_fp32(((block_q4_1_o *) dest)->d);
    float delta_expected = (src[30] - src[0]) / ((1 << 4) - 1);
    GGML_ASSERT(delta_result == delta_expected, "%f, %f", delta_result, delta_expected);

    float min_result = ggml_fp16_to_fp32(((block_q4_1_o *) dest)->m);
    float min_expected = src[0];
    GGML_ASSERT(min_result == min_expected, "%f, %f", min_result, min_expected);

    uint16_t outlier_index = ((block_q4_1_o *) dest)->outlier_index;
    uint16_t outlier_index_expected = 31;
    GGML_ASSERT(outlier_index == outlier_index_expected, "%d, %d", outlier_index, outlier_index_expected);

    float outlier_value_result = ggml_fp16_to_fp32(((block_q4_1_o *) dest)->outlier_value);
    float outlier_value_expected = src[31];
    GGML_ASSERT(outlier_value_result == outlier_value_expected, "%f, %f", outlier_value_result, outlier_value_expected);

    for (int i = 0; i < QK4_1_O - 1; i++) {
        uint8_t q4_result = (i % 2) ? (dest[sizeof(float) * 2 + i / 2] >> 4) : (dest[sizeof(float) * 2 + i / 2] & 0xF);
        uint8_t q4_expected = roundf((src[i] - min_expected) / delta_expected);
        GGML_ASSERT(q4_result == q4_expected, "%d: %d, %d", i, q4_result, q4_expected);
    }

    // --- Dequantization ---
    float dequantized[QK4_1_O];
    (quantize_fns.dequantize_row_q)(dest, dequantized, QK4_1_O);

    for (int i = 0; i < QK4_1_O; i++) {
        float actual = dequantized[i];
        float expected = src[i];
        float diff = fabsf(actual - expected);
        // Difference looks huge, but the range is 0..31 -- compared to the range, it is not that huge
        GGML_ASSERT(diff <= 1.0F, "%d: %f, %f", i, actual, expected);
    }

    // --- Matmul ---
    srand(42);

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, QK4_1_O, 4);

    for (int i = 0; i < QK4_1_O * 4; i++) {
        GGML_SET_ELEMENT_F32(mat, i, RANDOM_FLOAT());
    }

    struct ggml_tensor * quantized_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1_O, QK4_1_O, 4);

    int64_t histogram[16];

    ggml_quantize_q4_1_o(mat->data, quantized_mat->data, QK4_1_O * 4, QK4_1_O, histogram);

    struct ggml_tensor * vec = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, QK4_1_O);

    for (int i = 0; i < QK4_1_O; i++) {
        GGML_SET_ELEMENT_F32(vec, i, RANDOM_FLOAT());
    }

    struct ggml_tensor * expected_result = ggml_mul_mat(ctx, mat, vec);
    struct ggml_tensor * quantized_result = ggml_mul_mat(ctx, quantized_mat, vec);

    struct ggml_cgraph graph = ggml_build_forward(expected_result);
    ggml_build_forward_expand(&graph, quantized_result);
    graph.n_threads = 2;
    ggml_graph_compute(ctx, &graph);

    float diff_sum = 0.0F;

    for (int i = 0; i < 4; i++) {
        fprintf(
            stderr,
            "[%d] expected %f, actual %f\n",
            i,
            GGML_GET_ELEMENT_F32(expected_result, i),
            GGML_GET_ELEMENT_F32(quantized_result, i)
        );

        diff_sum += fabsf(GGML_GET_ELEMENT_F32(expected_result, i) - GGML_GET_ELEMENT_F32(quantized_result, i));
    }

    float diff_average = diff_sum / 4;

    GGML_ASSERT(diff_average <= 0.086357F, "Unexpected average difference value %f", diff_average);

    ggml_print_objects(ctx);

    ggml_free(ctx);

    return 0;
}
