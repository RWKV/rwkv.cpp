// Tests that tiny RWKV outputs expected results in all data types.
#include <stdlib.h>
#include <stdio.h>

#include <rwkv.h>

#include "logit_difference_validator.inc"

int main(void) {
    fprintf(stderr, "System info: %s\n", rwkv_get_system_info_string());

    // Silences the overly verbose output during quantization.
    rwkv_set_print_errors(NULL, false);

    float * expected_logits = calloc(N_VOCAB, sizeof(float));
    load_expected_logits(expected_logits);

    // Somehow when using cuBLAS the result of Q4_1 is different from CPU only.
    float expected_difference_sum[14] = {
        +0.000000F, // FP32
        -0.005320F, // FP16

        -0.160030F, // Q4_0
#if defined(GGML_USE_CUBLAS)
        -0.547409F, // Q4_1
#else
        -0.370606F, // Q4_1
#endif
        -0.170404F, // Q5_0
        +0.278034F, // Q5_1
        +0.071216F, // Q8_0

        +0.154614F, // Q4_0
#if defined(GGML_USE_CUBLAS)
        -0.539827F, // Q4_1
#else
        -0.372169F, // Q4_1
#endif
        -0.170043F, // Q5_0
        +0.294953F, // Q5_1
        +0.065571F, // Q8_0
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
