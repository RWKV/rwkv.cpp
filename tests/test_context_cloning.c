// Tests that evaluation works after the context was cloned.
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <rwkv.h>

#include "assertions.inc"

int main(void) {
    struct rwkv_context * ctx = rwkv_init_from_file("tiny-rwkv-5v2-730K-FP32.bin", 2, 0);

    ASSERT(ctx != NULL, "Unexpected error 0x%.8X", rwkv_get_last_error(NULL));

    float * state = calloc(rwkv_get_state_len(ctx), sizeof(float));
    float * logits = calloc(rwkv_get_logits_len(ctx), sizeof(float));

    ASSERT(state != NULL, "Failed to allocate state");
    ASSERT(logits != NULL, "Failed to allocate logits");

    const uint8_t prompt[12] = "hello world";

    rwkv_eval(ctx, prompt[0], NULL, state, logits);

    for (size_t i = 1; prompt[i] != 0; i++) {
        rwkv_eval(ctx, prompt[i], state, state, logits);
    }

    float * expected_logits = logits;

    logits = calloc(rwkv_get_logits_len(ctx), sizeof(float));

    ASSERT(logits != NULL, "Failed to allocate logits");

    struct rwkv_context * ctx2 = rwkv_clone_context(ctx, 2);

    ASSERT(ctx != ctx2, "Same context was returned");

    // The cloned context should work fine after the original context was freed.
    rwkv_free(ctx);

    rwkv_eval(ctx2, prompt[0], NULL, state, logits);

    for (int i = 1; prompt[i] != 0; i++) {
        rwkv_eval(ctx2, prompt[i], state, state, logits);
    }

    ASSERT(memcmp(expected_logits, logits, rwkv_get_logits_len(ctx2) * sizeof(float)) == 0, "Results are not identical");

    rwkv_free(ctx2);

    free(expected_logits);
    free(logits);
    free(state);

    return 0;
}
