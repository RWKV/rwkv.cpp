#include <rwkv.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define L() { fprintf(stderr, "L%d\n", __LINE__); }

int main() {
L();	struct rwkv_context * ctx = rwkv_init_from_file("tiny-rwkv-660K-FP32.bin", 2);
L();
L();	if (!ctx) {
L();		enum rwkv_error_flags error = rwkv_get_last_error(NULL);
L();		fprintf(stderr, "Unexpected error 0x%.8X\n", error);
L();		return EXIT_FAILURE;
L();	}
L();
L();	float * state = calloc(rwkv_get_state_buffer_element_count(ctx), sizeof(float));
L();	float * logits = calloc(rwkv_get_logits_buffer_element_count(ctx), sizeof(float));
L();
L();	if (!state || !logits) {
L();		fprintf(stderr, "Failed to allocate state/logits\n");
L();		return EXIT_FAILURE;
L();	}
L();
L();	// 0xd1 or 209 is space (0x20 or \u0120 in tokenizer)
L();	const unsigned char * prompt = "hello\xd1world";
L();
L();	rwkv_eval(ctx, prompt[0], NULL, state, logits);
L();
L();	for (const unsigned char * token = prompt + 1; *token != 0; token++) {
L();		rwkv_eval(ctx, *token, state, state, logits);
L();	}
L();
L();	float * expected_logits = logits;
L();	logits = calloc(rwkv_get_logits_buffer_element_count(ctx), sizeof(float));
L();
L();	if (!logits) {
L();		fprintf(stderr, "Failed to allocate state/logits\n");
L();		return EXIT_FAILURE;
L();	}
L();
L();	struct rwkv_context * ctx2 = rwkv_clone_context(ctx, 2);
L();
L();	rwkv_eval(ctx, prompt[0], NULL, state, logits);
L();
L();	for (const unsigned char * token = prompt + 1; *token != 0; token++) {
L();		rwkv_eval(ctx, *token, state, state, logits);
L();	}
L();
L();	if (memcmp(expected_logits, logits, rwkv_get_logits_buffer_element_count(ctx) * sizeof(float))) {
L();		fprintf(stderr, "results not identical :(\n");
L();		return EXIT_FAILURE;
L();	} else {
L();		fprintf(stdout, "Results identical, success!\n");
L();	}
L();
L();	rwkv_free(ctx);
L();	rwkv_free(ctx2);
L();
L();	free(expected_logits);
L();	free(logits);
L();	free(state);
L();
L();	return EXIT_SUCCESS;
}