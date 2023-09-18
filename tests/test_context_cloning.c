// Tests that evaluation works after the context was cloned.
#include <rwkv.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main() {
	struct rwkv_context * ctx = rwkv_init_from_file("tiny-rwkv-660K-FP32.bin", 2);

	if (!ctx) {
		enum rwkv_error_flags error = rwkv_get_last_error(NULL);
		fprintf(stderr, "Unexpected error 0x%.8X\n", error);
		return EXIT_FAILURE;
	}

	float * state = calloc(rwkv_get_state_len(ctx), sizeof(float));
	float * logits = calloc(rwkv_get_logits_len(ctx), sizeof(float));

	if (!state || !logits) {
		fprintf(stderr, "Failed to allocate state or logits\n");
		return EXIT_FAILURE;
	}

	const unsigned char prompt[12] = "hello world";

	rwkv_eval(ctx, prompt[0], NULL, state, logits);

	for (int i = 1; prompt[i] != 0; i++) {
		rwkv_eval(ctx, prompt[i], state, state, logits);
	}

	float * expected_logits = logits;

	logits = calloc(rwkv_get_logits_len(ctx), sizeof(float));

	if (!logits) {
		fprintf(stderr, "Failed to allocate logits\n");
		return EXIT_FAILURE;
	}

	struct rwkv_context * ctx2 = rwkv_clone_context(ctx, 2);

	if (ctx == ctx2) {
		fprintf(stderr, "Same context was returned\n");
		return EXIT_FAILURE;
	}

    // The cloned context should work fine after the original context was freed.
	rwkv_free(ctx);

	rwkv_eval(ctx2, prompt[0], NULL, state, logits);

	for (int i = 1; prompt[i] != 0; i++) {
		rwkv_eval(ctx2, prompt[i], state, state, logits);
	}

	if (memcmp(expected_logits, logits, rwkv_get_logits_len(ctx2) * sizeof(float))) {
		fprintf(stderr, "Results are not identical :(\n");
		return EXIT_FAILURE;
	} else {
		fprintf(stdout, "Results are identical, success!\n");
	}

	rwkv_free(ctx2);

	free(expected_logits);
	free(logits);
	free(state);

	return EXIT_SUCCESS;
}
