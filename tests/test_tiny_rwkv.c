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

void test_model(const char * model_path, const float * expected_logits, const float max_diff) {
    fprintf(stderr, "Testing %s\n", model_path);

    struct rwkv_context * model = rwkv_init_from_file(model_path, N_THREADS);

    uint32_t n_vocab = rwkv_get_logits_buffer_element_count(model);

    ASSERT(n_vocab == N_VOCAB, "Unexpected n_vocab in the model");

    float * state = malloc(sizeof(float) * rwkv_get_state_buffer_element_count(model));
    float * logits = malloc(sizeof(float) * n_vocab);

    char * prompt = "\"in";

    const size_t prompt_length = strlen(prompt);

    for (size_t i = 0; i < prompt_length; i++) {
        rwkv_eval(model, prompt[i], i == 0 ? NULL : state, state, logits);
    }

    float diff_sum = 0.0F;

    for (uint32_t i = 0; i < n_vocab; i++) {
        diff_sum += logits[i] - expected_logits[i];
    }

    fprintf(stderr, "Difference sum: %f\n", diff_sum);

    // When something breaks, difference would be way more than 10
    ASSERT(fabsf(diff_sum) <= fabsf(max_diff) + 0.01F, "Too big difference %f, expected no more than %f", diff_sum, max_diff);

    rwkv_free(model);

    free(state);
    free(logits);
}

int main(int argc, const char ** argv) {
    fprintf(stderr, "System info: %s\n", rwkv_get_system_info_string());

    float * expected_logits = malloc(sizeof(float) * N_VOCAB);
    FILE * file = fopen("expected_logits.bin", "rb");
    ASSERT(file != NULL, "Failed to open expected_logits.bin");
    size_t elements_read = fread(expected_logits, sizeof(float), N_VOCAB, file);
    ASSERT(elements_read == N_VOCAB, "Failed to read expected_logits.bin, read %zd elements", elements_read);
    fclose(file);

    float expected_difference_sum[12] = {
        0.000000F,
        -0.005320F,

        -0.501214F,
        -0.370606F,
        -0.268956F,
        0.676837F,
        0.237099F,

        -0.501073F,
        -0.372169F,
        -0.244590F,
        0.674874F,
        0.243007F
    };

    test_model("tiny-rwkv-660K-FP32.bin", expected_logits, expected_difference_sum[0]);
    test_model("tiny-rwkv-660K-FP16.bin", expected_logits, expected_difference_sum[1]);

    rwkv_quantize_model_file("tiny-rwkv-660K-FP32.bin", "tiny-rwkv-660K-FP32-Q4_0.bin", 2);
    rwkv_quantize_model_file("tiny-rwkv-660K-FP32.bin", "tiny-rwkv-660K-FP32-Q4_1.bin", 3);
    rwkv_quantize_model_file("tiny-rwkv-660K-FP32.bin", "tiny-rwkv-660K-FP32-Q4_1_O.bin", 4);
    rwkv_quantize_model_file("tiny-rwkv-660K-FP32.bin", "tiny-rwkv-660K-FP32-Q4_2.bin", 5);
    rwkv_quantize_model_file("tiny-rwkv-660K-FP32.bin", "tiny-rwkv-660K-FP32-Q4_3.bin", 6);

    test_model("tiny-rwkv-660K-FP32-Q4_0.bin", expected_logits, expected_difference_sum[2]);
    test_model("tiny-rwkv-660K-FP32-Q4_1.bin", expected_logits, expected_difference_sum[3]);
    test_model("tiny-rwkv-660K-FP32-Q4_1_O.bin", expected_logits, expected_difference_sum[4]);
    test_model("tiny-rwkv-660K-FP32-Q4_2.bin", expected_logits, expected_difference_sum[5]);
    test_model("tiny-rwkv-660K-FP32-Q4_3.bin", expected_logits, expected_difference_sum[6]);

    rwkv_quantize_model_file("tiny-rwkv-660K-FP16.bin", "tiny-rwkv-660K-FP16-Q4_0.bin", 2);
    rwkv_quantize_model_file("tiny-rwkv-660K-FP16.bin", "tiny-rwkv-660K-FP16-Q4_1.bin", 3);
    rwkv_quantize_model_file("tiny-rwkv-660K-FP16.bin", "tiny-rwkv-660K-FP16-Q4_1_O.bin", 4);
    rwkv_quantize_model_file("tiny-rwkv-660K-FP16.bin", "tiny-rwkv-660K-FP16-Q4_2.bin", 5);
    rwkv_quantize_model_file("tiny-rwkv-660K-FP16.bin", "tiny-rwkv-660K-FP16-Q4_3.bin", 6);

    test_model("tiny-rwkv-660K-FP16-Q4_0.bin", expected_logits, expected_difference_sum[7]);
    test_model("tiny-rwkv-660K-FP16-Q4_1.bin", expected_logits, expected_difference_sum[8]);
    test_model("tiny-rwkv-660K-FP16-Q4_1_O.bin", expected_logits, expected_difference_sum[9]);
    test_model("tiny-rwkv-660K-FP16-Q4_2.bin", expected_logits, expected_difference_sum[10]);
    test_model("tiny-rwkv-660K-FP16-Q4_3.bin", expected_logits, expected_difference_sum[11]);

    free(expected_logits);

    return 0;
}
