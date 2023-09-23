#ifndef LOGIT_DIFFERENCE_VALIDATOR_INC
#define LOGIT_DIFFERENCE_VALIDATOR_INC

#include <string.h>
#include <math.h>

#include <rwkv.h>

#include "assertions.inc"

// RWKV Tiny is a byte-level model.
#define N_VOCAB 256
// Also test multithreading.
#define N_THREADS 2

void load_expected_logits(float * expected_logits) {
    FILE * file = fopen("expected_logits.bin", "rb");
    ASSERT(file != NULL, "Failed to open expected_logits.bin");
    size_t elements_read = fread(expected_logits, sizeof(float), N_VOCAB, file);
    ASSERT(elements_read == N_VOCAB, "Failed to read expected_logits.bin, read %zd elements", elements_read);
    fclose(file);
}

void test_model(const char * model_path, const float * expected_logits, const float max_diff) {
    fprintf(stderr, "Testing %s\n", model_path);

    struct rwkv_context * model = rwkv_init_from_file(model_path, N_THREADS);
    enum rwkv_error_flags error = rwkv_get_last_error(NULL);
    ASSERT(error == 0, "Unexpected error %d", error);

#if defined(GGML_USE_CUBLAS)
    ASSERT(rwkv_gpu_offload_layers(model, rwkv_get_n_layer(model) + 1), "Failed to offload layers to GPU");
#endif

    const size_t n_vocab = rwkv_get_logits_len(model);

    ASSERT(n_vocab == N_VOCAB, "Unexpected n_vocab in the model");

    float * state = calloc(rwkv_get_state_len(model), sizeof(float));
    float * logits = calloc(n_vocab, sizeof(float));

    ASSERT(state != NULL, "Failed to allocate state");
    ASSERT(logits != NULL, "Failed to allocate logits");

    const char * prompt = "\"in";
    const uint32_t prompt_seq[] = { '"', 'i', 'n' };
    const size_t prompt_length = strlen(prompt);

    // ---

    rwkv_init_state(model, state);

    for (size_t i = 0; prompt[i] != 0; i++) {
        rwkv_eval(model, prompt[i], state, state, logits);
    }

    float diff_sum = 0.0F;

    for (uint32_t i = 0; i < n_vocab; i++) {
        diff_sum += logits[i] - expected_logits[i];
    }

    fprintf(stderr, "Serial difference sum: %f\n", diff_sum);

    // When something breaks, difference would be way more than 10
    ASSERT(fabsf(diff_sum) <= fabsf(max_diff) + 0.01F, "Too big serial difference %f, expected no more than %f", (double) diff_sum, (double) max_diff);

    // ---

    rwkv_init_state(model, state);
    rwkv_eval_sequence(model, prompt_seq, prompt_length, state, state, logits);

    diff_sum = 0.0F;

    for (uint32_t i = 0; i < n_vocab; i++) {
        diff_sum += logits[i] - expected_logits[i];
    }

    fprintf(stderr, "Sequence difference sum: %f\n", diff_sum);

    // When something breaks, difference would be way more than 10
    ASSERT(fabsf(diff_sum) <= fabsf(max_diff) + 0.01F, "Too big sequence difference %f, expected no more than %f", (double) diff_sum, (double) max_diff);

    // ---

    rwkv_free(model);

    free(state);
    free(logits);
}

#endif
