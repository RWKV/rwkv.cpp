// Tests that tiny RWKV outputs expected results in all data types.

#include "ggml.h"
#include "rwkv.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ASSERT(x, ...) {\
        if (!(x)) {\
            fprintf(stderr, "*** Assertion failed ***\n");\
            fprintf(stderr, __VA_ARGS__);\
            fprintf(stderr, "\n%s:%d\n", __FILE__, __LINE__);\
            abort();\
        }\
    }

// ---

#define N_VOCAB 256
#define N_THREADS 2
#define N_GPU_LAYERS 1

void test_model(const char * model_path, const float * expected_logits, const float max_diff) {
    fprintf(stderr, "Testing %s\n", model_path);

    struct rwkv_context * model = rwkv_init_from_file(model_path, N_THREADS);
    enum rwkv_error_flags error = rwkv_get_last_error(NULL);
    ASSERT(model && error == 0, "Unexpected error %d", error);
    struct rwkv_context * model2 = rwkv_clone_context(model, N_THREADS);
    enum rwkv_error_flags error2 = rwkv_get_last_error(NULL);
    ASSERT(model2 && error2 == 0, "Unexpected error2 %d", error2);
#ifdef GGML_USE_CUBLAS
    ASSERT(rwkv_gpu_offload_layers(model, N_GPU_LAYERS), "Unexpected error %d", rwkv_get_last_error(model));
#endif

    uint32_t n_vocab = rwkv_get_logits_buffer_element_count(model);

    ASSERT(n_vocab == N_VOCAB, "Unexpected n_vocab in the model");

    float * state = malloc(sizeof(float) * rwkv_get_state_buffer_element_count(model));
    float * state2 = malloc(sizeof(float) * rwkv_get_state_buffer_element_count(model2));
    float * logits = malloc(sizeof(float) * n_vocab);
    float * logits2 = malloc(sizeof(float) * n_vocab);

    char * prompt = "\"in";

    const size_t prompt_length = strlen(prompt);

    for (size_t i = 0; i < prompt_length; i++) {
        rwkv_eval(model, prompt[i], i == 0 ? NULL : state, state, logits);
        rwkv_eval(model2, prompt[i], i == 0 ? NULL : state2, state2, logits2);
    }

    float diff_sum = 0.0F;
    float diff_sum2 = 0.0F;

    for (uint32_t i = 0; i < n_vocab; i++) {
        diff_sum += logits[i] - expected_logits[i];
        diff_sum2 += logits2[i] - expected_logits[i];
    }

    fprintf(stderr, "Difference sum: %f\n", diff_sum);
    fprintf(stderr, "Difference sum2: %f\n", diff_sum2);

    // When something breaks, difference would be way more than 10
    ASSERT(fabsf(diff_sum) <= fabsf(max_diff) + 0.01F, "Too big difference %f, expected no more than %f", (double) diff_sum, (double) max_diff);
    ASSERT(fabsf(diff_sum2) <= fabsf(max_diff) + 0.01F, "Too big difference2 %f, expected no more than %f", (double) diff_sum2, (double) max_diff);

    rwkv_free(model);
    rwkv_free(model2);

    free(state);
    free(state2);
    free(logits);
    free(logits2);
}

int main(void) {
    fprintf(stderr, "System info: %s\n", rwkv_get_system_info_string());

    float * expected_logits = malloc(sizeof(float) * N_VOCAB);
    FILE * file = fopen("expected_logits.bin", "rb");
    ASSERT(file != NULL, "Failed to open expected_logits.bin");
    size_t elements_read = fread(expected_logits, sizeof(float), N_VOCAB, file);
    ASSERT(elements_read == N_VOCAB, "Failed to read expected_logits.bin, read %zd elements", elements_read);
    fclose(file);

    // Somehow when using cuBLAS the calculation of Q4_1 may different from cpu only
    float expected_difference_sum[14] = {
        0.000000F,
        -0.005320F,

        -0.160030F,
#ifdef GGML_USE_CUBLAS
        -0.412408F,
#else
        -0.370606F,
#endif
        -0.170404F,
        0.278034F,
        0.071216F,

        0.154614F,
#ifdef GGML_USE_CUBLAS
        -0.405527F,
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
