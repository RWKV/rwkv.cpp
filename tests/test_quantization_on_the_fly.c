// Tests that results from on-the-fly quantized model are identical with results of pre-quantized model.

#include "ggml.h"
#include "rwkv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_THREADS 2

int main(void) {
    rwkv_quantize_model_file("tiny-rwkv-660K-FP32.bin", "tiny-rwkv-660K-FP32-Q5_1.bin", "Q5_1");

	struct rwkv_context * prequantized_ctx = rwkv_init_from_file("tiny-rwkv-660K-FP32-Q5_1.bin", N_THREADS);

	if (!prequantized_ctx) {
		enum rwkv_error_flags error = rwkv_get_last_error(NULL);
		fprintf(stderr, "Unexpected error 0x%.8X\n", error);
		return EXIT_FAILURE;
	}

    // ---

    struct rwkv_init_from_file_option option = {RWKV_INIT_FROM_FILE_OPTION_TARGET_FORMAT_NAME, "Q5_1"};

	struct rwkv_context * on_the_fly_quantized_ctx = rwkv_init_from_file_ex("tiny-rwkv-660K-FP32.bin", N_THREADS, &option, 1);

	if (!on_the_fly_quantized_ctx) {
		enum rwkv_error_flags error = rwkv_get_last_error(NULL);
		fprintf(stderr, "Unexpected error 0x%.8X\n", error);
		return EXIT_FAILURE;
	}

	// ---

	float * state = calloc(rwkv_get_state_len(prequantized_ctx), sizeof(float));

	if (!state) {
		fprintf(stderr, "Failed to allocate state\n");
		return EXIT_FAILURE;
	}

	float * expected_logits = calloc(rwkv_get_logits_len(prequantized_ctx), sizeof(float));

	if (!expected_logits) {
		fprintf(stderr, "Failed to allocate logits\n");
		return EXIT_FAILURE;
	}

	const unsigned char prompt[12] = "hello world";

	rwkv_eval(prequantized_ctx, prompt[0], NULL, state, expected_logits);

	for (int i = 1; prompt[i] != 0; i++) {
		rwkv_eval(prequantized_ctx, prompt[i], state, state, expected_logits);
	}

	// ---

	float * actual_logits = calloc(rwkv_get_logits_len(on_the_fly_quantized_ctx), sizeof(float));

	if (!actual_logits) {
		fprintf(stderr, "Failed to allocate logits\n");
		return EXIT_FAILURE;
	}

	rwkv_eval(on_the_fly_quantized_ctx, prompt[0], NULL, state, actual_logits);

	for (int i = 1; prompt[i] != 0; i++) {
		rwkv_eval(on_the_fly_quantized_ctx, prompt[i], state, state, actual_logits);
	}

	// ---

	if (memcmp(expected_logits, actual_logits, rwkv_get_logits_len(on_the_fly_quantized_ctx) * sizeof(float))) {
		fprintf(stderr, "Results not identical :(\n");
		return EXIT_FAILURE;
	} else {
		fprintf(stdout, "Results identical, success!\n");
	}

	rwkv_free(on_the_fly_quantized_ctx);
	rwkv_free(prequantized_ctx);

	free(expected_logits);
	free(actual_logits);
	free(state);

    return 0;
}
