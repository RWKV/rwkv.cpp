// Tests that eval_sequence_in_chunks gives results equivalent to serial eval.
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <rwkv.h>

#include "assertions.inc"

void test_on_prompt(const char * prompt, const size_t prompt_length) {
    fprintf(stderr, "Calculating expected state and logits for prompt of size %zd\n", prompt_length);

    struct rwkv_context * ctx = rwkv_init_from_file("tiny-rwkv-5v2-730K-FP32.bin", 2, 0);

    ASSERT(ctx != NULL, "Unexpected error 0x%.8X", rwkv_get_last_error(NULL));

    float * expected_state = calloc(rwkv_get_state_len(ctx), sizeof(float));
    float * expected_logits = calloc(rwkv_get_logits_len(ctx), sizeof(float));

    ASSERT(expected_state != NULL, "Failed to allocate state");
    ASSERT(expected_logits != NULL, "Failed to allocate logits");

    rwkv_eval(ctx, prompt[0], NULL, expected_state, expected_logits);

    for (size_t i = 1; prompt[i] != 0; i++) {
        rwkv_eval(ctx, prompt[i], expected_state, expected_state, expected_logits);
    }

    // ---

    uint32_t * prompt_tokens = calloc(prompt_length, sizeof(uint32_t));

    for (int i = 0; i < prompt_length; i++) {
        prompt_tokens[i] = prompt[i];
    }

    // ---

    float * state = calloc(rwkv_get_state_len(ctx), sizeof(float));
    float * logits = calloc(rwkv_get_logits_len(ctx), sizeof(float));

    ASSERT(state != NULL, "Failed to allocate state");
    ASSERT(logits != NULL, "Failed to allocate logits");

    const size_t chunk_sizes[4] = {1, 2, 8, 10};

    for (int i = 0; i < 4; i++) {
        size_t chunk_size = chunk_sizes[i];

        fprintf(stderr, "Testing chunk_size = %zd\n", chunk_size);

        rwkv_eval_sequence_in_chunks(ctx, prompt_tokens, prompt_length, chunk_size, NULL, state, logits);

        ASSERT(memcmp(expected_state, state, rwkv_get_state_len(ctx) * sizeof(float)) == 0, "Results are not identical");
    }

    // ---

    rwkv_free(ctx);

    free(logits);
    free(state);
    free(expected_logits);
    free(expected_state);
    free(prompt_tokens);
}

int main(void) {
    const char prompt1[70 + 1] = "This is a port of [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM";
    test_on_prompt(prompt1, 70);

    const char prompt2[1 + 1] = "T";
    test_on_prompt(prompt2, 1);

    return 0;
}
