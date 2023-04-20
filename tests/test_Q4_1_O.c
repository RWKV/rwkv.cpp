// Tests that Q4_1_O basics (quantization, dequantization, matmul) work.

#include "ggml.h"
#include "rwkv.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GET_ELEMENT_F32(tensor, i) (((float *) tensor->data)[i])

#define SET_ELEMENT_F32(tensor, i, value) ((float *) tensor->data)[i] = value

#define ASSERT(x, ...) {\
        if (!(x)) {\
            fprintf(stderr, "*** Assertion failed ***\n");\
            fprintf(stderr, __VA_ARGS__);\
            fprintf(stderr, "\n%s:%d\n", __FILE__, __LINE__);\
            abort();\
        }\
    }

// ---

#define QK 32

// Copied from ggml.c
typedef struct {
    ggml_fp16_t d;
    ggml_fp16_t m;
    uint16_t outlier_index;
    ggml_fp16_t outlier_value;
    uint8_t qs[QK / 2];
} block_q4_1_o;

int main(int argc, const char ** argv) {
    ASSERT(sizeof(block_q4_1_o) == 8 + QK / 2, "Wrong q4_1_o block size/padding");

    // Needed to initialize FP16 lookup table
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    fprintf(stderr, "System info: %s\n", rwkv_get_system_info_string());

    quantize_fns_t quantize_fns = ggml_internal_get_quantize_fn(GGML_TYPE_Q4_1_O);

    float src[QK];
    uint8_t dest[24];

    // 1..32
    for (int i = 0; i < QK; i++) {
        src[i] = (float) (i + 1);
    }

    // --- Quantization ---
    (quantize_fns.quantize_row_q)(src, dest, QK);

    float delta_result = ggml_fp16_to_fp32(((block_q4_1_o *) dest)->d);
    float delta_expected = (src[30] - src[0]) / ((1 << 4) - 1);
    ASSERT(delta_result == delta_expected, "%f, %f", delta_result, delta_expected);

    float min_result = ggml_fp16_to_fp32(((block_q4_1_o *) dest)->m);
    float min_expected = src[0];
    ASSERT(min_result == min_expected, "%f, %f", min_result, min_expected);

    uint16_t outlier_index = ((block_q4_1_o *) dest)->outlier_index;
    uint16_t outlier_index_expected = 31;
    ASSERT(outlier_index == outlier_index_expected, "%d, %d", outlier_index, outlier_index_expected);

    float outlier_value_result = ggml_fp16_to_fp32(((block_q4_1_o *) dest)->outlier_value);
    float outlier_value_expected = src[31];
    ASSERT(outlier_value_result == outlier_value_expected, "%f, %f", outlier_value_result, outlier_value_expected);

    for (int i = 0; i < QK - 1; i++) {
        uint8_t q4_result = (i % 2) ? (dest[sizeof(float) * 2 + i / 2] >> 4) : (dest[sizeof(float) * 2 + i / 2] & 0xF);
        uint8_t q4_expected = roundf((src[i] - min_expected) / delta_expected);
        ASSERT(q4_result == q4_expected, "%d: %d, %d", i, q4_result, q4_expected);
    }

    // --- Dequantization ---
    float dequantized[QK];
    (quantize_fns.dequantize_row_q)(dest, dequantized, QK);

    for (int i = 0; i < QK; i++) {
        float actual = dequantized[i];
        float expected = src[i];
        float diff = fabsf(actual - expected);
        // Difference looks huge, but the range is 0..31 -- compared to the range, it is not that huge
        ASSERT(diff <= 1.0F, "%d: %f, %f", i, actual, expected);
    }

    // --- Matmul ---
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, QK, 4);

    // Note rare outlier values: -88, -83, etc.
    float mat_values[QK * 4] = {
        -1.371795F, -88.901100F, -0.412088F, -0.486081F, 1.280220F, -1.067033F, 1.371795F, 1.099267F, 1.079487F, -0.204029F, 1.237729F, -0.563736F,
        -0.633333F, 0.700000F, 0.211355F, 0.510989F, -0.981319F, -0.456777F, 0.011355F, 0.911722F, -0.976191F, 0.078022F, -0.757143F, -0.744689F,
        -0.768865F, 0.656777F, 0.141026F, -0.038462F, 1.023810F, 1.221612F, -0.393773F, 1.135165F, -1.341758F, -83.113556F, 1.291209F, 0.313187F,
        1.032601F, -0.401099F, 1.482418F, 0.823077F, 0.619414F, -0.583516F, 0.527106F, 1.489011F, 1.327839F, 0.846520F, -1.437729F, 0.461172F,
        1.031136F, 0.293407F, 0.284615F, -1.102198F, -1.481685F, 0.602564F, -0.480952F, -0.745421F, -1.376190F, -1.319780F, 1.338828F, -1.062637F,
        1.266300F, 0.360073F, 1.472894F, 1.063370F, -0.833333F, 49.047626F, -1.229670F, 1.079487F, -0.004762F, -0.696337F, -0.541758F, 0.993773F,
        -1.323443F, 0.908059F, -1.059707F, 0.965201F, -0.376923F, 1.158608F, -1.100000F, -1.002564F, -0.355678F, 1.157143F, 0.450916F, -0.497802F,
        1.270696F, 0.028205F, 1.075092F, 1.462637F, 0.252381F, -0.579121F, -0.880220F, -0.041392F, -1.017949F, -0.754945F, 0.582784F, -1.193773F,
        -1.411355F, 122.014656F, -1.053114F, -0.949084F, 0.448718F, 0.209890F, 0.815751F, 0.071429F, -0.125641F, -0.600366F, -0.914652F, -0.956410F,
        -0.278755F, 0.235531F, -0.573260F, -1.484615F, -0.327839F, -0.297070F, -1.195238F, -1.160073F, 0.932967F, -0.606960F, 0.798901F, 0.212088F,
        0.113187F, -0.116117F, -0.532967F, 0.077289F, 0.016484F, 1.352747F, -1.487546F, -1.363736F
    };

    for (int i = 0; i < QK * 4; i++) {
        SET_ELEMENT_F32(mat, i, mat_values[i]);
    }

    struct ggml_tensor * quantized_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_1_O, QK, 4);

    int64_t histogram[16];

    ggml_quantize_q4_1_o(mat->data, quantized_mat->data, QK * 4, QK, histogram);

    struct ggml_tensor * vec = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, QK);

    float vec_values[] = {
        -0.578388F, -0.770330F, -0.183516F, 0.264103F, 0.585714F, -0.226740F, 1.319048F, 0.652381F,
        -1.161538F, 0.428205F, -0.907326F, -0.837729F, 0.673626F, 0.248718F, 0.392308F, -0.225275F,
        0.910989F, 0.483150F, -0.669963F, -0.412088F, 0.954945F, 0.826007F, 0.113919F, 0.095604F,
        -1.042125F, -1.094872F, 0.589377F, -0.426007F, 0.669231F, -0.243590F, -0.179121F, 0.325641F
    };

    for (int i = 0; i < QK; i++) {
        SET_ELEMENT_F32(vec, i, vec_values[i]);
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
            GET_ELEMENT_F32(expected_result, i),
            GET_ELEMENT_F32(quantized_result, i)
        );

        diff_sum += fabsf(GET_ELEMENT_F32(expected_result, i) - GET_ELEMENT_F32(quantized_result, i));
    }

    float diff_average = diff_sum / 4;

    // If Q4_1_O format works correctly, difference should be this or lower
    ASSERT(diff_average <= 0.112F, "Unexpected average difference value %f", diff_average);

    ggml_free(ctx);

    return 0;
}
