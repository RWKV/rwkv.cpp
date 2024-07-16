// Tests that evaluation works when the logits parameter was set to NULL.
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <rwkv.h>

#include "assertions.inc"

#define TOKEN_COUNT 11

const char prompt[TOKEN_COUNT + 1] = "hello world";

void test_serial_mode(void) {
    fprintf(stderr, "Testing serial mode\n");

    struct rwkv_context * ctx = rwkv_init_from_file("tiny-rwkv-5v2-730K-FP32.bin", 2, 0);

    ASSERT(ctx != NULL, "Unexpected error 0x%.8X", rwkv_get_last_error(NULL));

    float * state = calloc(rwkv_get_state_len(ctx), sizeof(float));
    float * logits = calloc(rwkv_get_logits_len(ctx), sizeof(float));

    ASSERT(state != NULL, "Failed to allocate state");
    ASSERT(logits != NULL, "Failed to allocate logits");

    rwkv_eval(ctx, prompt[0], NULL, state, logits);

    for (size_t i = 1; prompt[i] != 0; i++) {
        rwkv_eval(ctx, prompt[i], state, state, logits);
    }

    float * expected_state = state;

    state = calloc(rwkv_get_state_len(ctx), sizeof(float));

    ASSERT(state != NULL, "Failed to allocate state");

    rwkv_eval(ctx, prompt[0], NULL, state, NULL);

    for (int i = 1; prompt[i] != 0; i++) {
        rwkv_eval(ctx, prompt[i], state, state, NULL);
    }

    ASSERT(memcmp(expected_state, state, rwkv_get_state_len(ctx) * sizeof(float)) == 0, "Results are not identical");

    rwkv_free(ctx);

    free(logits);
    free(state);
    free(expected_state);
}

void test_sequential_mode(void) {
    fprintf(stderr, "Testing sequential mode\n");

    struct rwkv_context * ctx = rwkv_init_from_file("tiny-rwkv-5v2-730K-FP32.bin", 2, 0);

    ASSERT(ctx != NULL, "Unexpected error 0x%.8X", rwkv_get_last_error(NULL));

    float * state = calloc(rwkv_get_state_len(ctx), sizeof(float));
    float * logits = calloc(rwkv_get_logits_len(ctx), sizeof(float));

    ASSERT(state != NULL, "Failed to allocate state");
    ASSERT(logits != NULL, "Failed to allocate logits");

    uint32_t prompt_tokens[TOKEN_COUNT];

    for (int i = 0; i < TOKEN_COUNT; i++) {
        prompt_tokens[i] = prompt[i];
    }

    rwkv_eval_sequence(ctx, prompt_tokens, TOKEN_COUNT, NULL, state, logits);

    float * expected_state = state;

    state = calloc(rwkv_get_state_len(ctx), sizeof(float));

    ASSERT(state != NULL, "Failed to allocate state");

    rwkv_eval_sequence(ctx, prompt_tokens, TOKEN_COUNT, NULL, state, NULL);

    ASSERT(memcmp(expected_state, state, rwkv_get_state_len(ctx) * sizeof(float)) == 0, "Results are not identical");

    rwkv_free(ctx);

    free(logits);
    free(state);
    free(expected_state);
}

int main(void) {
    test_serial_mode();

    test_sequential_mode();

    return 0;
}
