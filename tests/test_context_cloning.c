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
		fprintf(stderr, "Failed to allocate state/logits\n");
		return EXIT_FAILURE;
	}

	// 0xd1 or 209 is space (0x20 or \u0120 in tokenizer)
	const unsigned char * prompt = "hello\xd1world";

	rwkv_eval(ctx, prompt[0], NULL, state, logits);

	for (const unsigned char * token = prompt + 1; *token != 0; token++) {
		rwkv_eval(ctx, *token, state, state, logits);
	}

	float * expected_logits = logits;
	logits = calloc(rwkv_get_logits_len(ctx), sizeof(float));

	if (!logits) {
		fprintf(stderr, "Failed to allocate state/logits\n");
		return EXIT_FAILURE;
	}

	struct rwkv_context * ctx2 = rwkv_clone_context(ctx, 2);

	rwkv_eval(ctx, prompt[0], NULL, state, logits);

	for (const unsigned char * token = prompt + 1; *token != 0; token++) {
		rwkv_eval(ctx, *token, state, state, logits);
	}

	if (memcmp(expected_logits, logits, rwkv_get_logits_len(ctx) * sizeof(float))) {
		fprintf(stderr, "results not identical :(\n");
		return EXIT_FAILURE;
	} else {
		fprintf(stdout, "Results identical, success!\n");
	}

	rwkv_free(ctx);
	rwkv_free(ctx2);

	free(expected_logits);
	free(logits);
	free(state);

	return EXIT_SUCCESS;
}