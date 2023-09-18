// Tests that evaluation works when the logits parameter was set to NULL.
#include <rwkv.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int test_serial_mode() {
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

	float * expected_state = state;

	state = calloc(rwkv_get_state_len(ctx), sizeof(float));

	if (!state) {
		fprintf(stderr, "Failed to allocate state\n");
		return EXIT_FAILURE;
	}

	rwkv_eval(ctx, prompt[0], NULL, state, NULL);

	for (int i = 1; prompt[i] != 0; i++) {
		rwkv_eval(ctx, prompt[i], state, state, NULL);
	}

	if (memcmp(expected_state, state, rwkv_get_state_len(ctx) * sizeof(float))) {
		fprintf(stderr, "Serial mode: results are not identical :(\n");
		return EXIT_FAILURE;
	} else {
		fprintf(stdout, "Serial mode: results are identical, success!\n");
	}

	rwkv_free(ctx);

	free(logits);
	free(state);
	free(expected_state);

	return EXIT_SUCCESS;
}

static int test_sequential_mode() {
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

	rwkv_eval_sequence(ctx, prompt, 11, NULL, state, logits);

	float * expected_state = state;

	state = calloc(rwkv_get_state_len(ctx), sizeof(float));

	if (!state) {
		fprintf(stderr, "Failed to allocate state\n");
		return EXIT_FAILURE;
	}

	rwkv_eval_sequence(ctx, prompt, 11, NULL, state, NULL);

	if (memcmp(expected_state, state, rwkv_get_state_len(ctx) * sizeof(float))) {
		fprintf(stderr, "Sequential mode: results are not identical :(\n");
		return EXIT_FAILURE;
	} else {
		fprintf(stdout, "Sequential mode: results are identical, success!\n");
	}

	rwkv_free(ctx);

	free(logits);
	free(state);
	free(expected_state);

	return EXIT_SUCCESS;
}

int main() {
	int result = test_serial_mode();

	if (result != EXIT_SUCCESS) {
	    return result;
	}

	result = test_sequential_mode();

	if (result != EXIT_SUCCESS) {
	    return result;
	}

	return EXIT_SUCCESS;
}
