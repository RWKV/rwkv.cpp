// Tests that tiny RWKV outputs expected results in all data types.
#include <rwkv.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "logit_difference_validator.inc"

int main(void) {
    fprintf(stderr, "System info: %s\n", rwkv_get_system_info_string());

    float * expected_logits = malloc(sizeof(float) * N_VOCAB);
    load_expected_logits(expected_logits);

    // Somehow when using cuBLAS the calculation of Q4_1 may different from cpu only
    float expected_difference_sum[14] = {
        0.000000F,
        -0.005320F,

        -0.160030F,
#ifdef GGML_USE_CUBLAS
        -0.547409F,
#else
        -0.370606F,
#endif
        -0.170404F,
        0.278034F,
        0.071216F,

        0.154614F,
#ifdef GGML_USE_CUBLAS
        -0.539827F,
#else
        -0.372169F,
#endif
        -0.170043F,
        0.294953F,
        0.065571F,
    };

    test_model("tiny-rwkv-660K-FP32.bin", expected_logits, expected_difference_sum[0]);
    test_model("tiny-rwkv-660K-FP16.bin", expected_logits, expected_difference_sum[1]);

    rwkv_quantize_model_file("tiny-rwkv-660K-FP32.bin", "tiny-rwkv-660K-FP32-Q4_0.bin", "Q4_0");
    rwkv_quantize_model_file("tiny-rwkv-660K-FP32.bin", "tiny-rwkv-660K-FP32-Q4_1.bin", "Q4_1");
    rwkv_quantize_model_file("tiny-rwkv-660K-FP32.bin", "tiny-rwkv-660K-FP32-Q5_0.bin", "Q5_0");
    rwkv_quantize_model_file("tiny-rwkv-660K-FP32.bin", "tiny-rwkv-660K-FP32-Q5_1.bin", "Q5_1");
    rwkv_quantize_model_file("tiny-rwkv-660K-FP32.bin", "tiny-rwkv-660K-FP32-Q8_0.bin", "Q8_0");

    test_model("tiny-rwkv-660K-FP32-Q4_0.bin", expected_logits, expected_difference_sum[2]);
    test_model("tiny-rwkv-660K-FP32-Q4_1.bin", expected_logits, expected_difference_sum[3]);
    test_model("tiny-rwkv-660K-FP32-Q5_0.bin", expected_logits, expected_difference_sum[4]);
    test_model("tiny-rwkv-660K-FP32-Q5_1.bin", expected_logits, expected_difference_sum[5]);
    test_model("tiny-rwkv-660K-FP32-Q8_0.bin", expected_logits, expected_difference_sum[6]);

    rwkv_quantize_model_file("tiny-rwkv-660K-FP16.bin", "tiny-rwkv-660K-FP16-Q4_0.bin", "Q4_0");
    rwkv_quantize_model_file("tiny-rwkv-660K-FP16.bin", "tiny-rwkv-660K-FP16-Q4_1.bin", "Q4_1");
    rwkv_quantize_model_file("tiny-rwkv-660K-FP16.bin", "tiny-rwkv-660K-FP16-Q5_0.bin", "Q5_0");
    rwkv_quantize_model_file("tiny-rwkv-660K-FP16.bin", "tiny-rwkv-660K-FP16-Q5_1.bin", "Q5_1");
    rwkv_quantize_model_file("tiny-rwkv-660K-FP16.bin", "tiny-rwkv-660K-FP16-Q8_0.bin", "Q8_0");

    test_model("tiny-rwkv-660K-FP16-Q4_0.bin", expected_logits, expected_difference_sum[7]);
    test_model("tiny-rwkv-660K-FP16-Q4_1.bin", expected_logits, expected_difference_sum[8]);
    test_model("tiny-rwkv-660K-FP16-Q5_0.bin", expected_logits, expected_difference_sum[9]);
    test_model("tiny-rwkv-660K-FP16-Q5_1.bin", expected_logits, expected_difference_sum[10]);
    test_model("tiny-rwkv-660K-FP16-Q8_0.bin", expected_logits, expected_difference_sum[11]);

    free(expected_logits);

    return 0;
}
