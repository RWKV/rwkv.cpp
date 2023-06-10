#include "ggml.h"
#include "rwkv.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#ifdef _WIN32
bool QueryPerformanceFrequency(uint64_t* lpFrequency);
bool QueryPerformanceCounter(uint64_t* lpPerformanceCount);

#define time_t uint64_t
#define time_calibrate(freq) do { QueryPerformanceFrequency(&freq); freq /= 1000; } while (0)
#define time_measure(x) QueryPerformanceCounter(&x)
#define TIME_DIFF(freq, start, end) (double) ((end - start) / freq) / 1000.
#else
#include <time.h>

#define time_t struct timespec
#define time_calibrate(freq) (void) freq
#define time_measure(x) clock_gettime(CLOCK_MONOTONIC, &x)
#define TIME_DIFF(freq, start, end) (double) ((end.tv_nsec - start.tv_nsec) / 1000000) / 1000
#endif

int main() {
    time_t freq;
    time_calibrate(freq);

    rwkv_set_print_errors(NULL, false);

    fprintf(stderr, "%s\n\n", rwkv_get_system_info_string());

    time_t start_load, end_load;
    fprintf(stderr, "Load model ...");
    time_measure(start_load);
    struct rwkv_context * ctx = rwkv_init_from_file("C:\\Users\\LoganDark\\Documents\\RWKV\\fp32-rwkv-4-world-0.4b-v1-20230529-ctx4096.bin", 1);
    time_measure(end_load);
    fprintf(stderr, " %.3fs\n", TIME_DIFF(freq, start_load, end_load));

    time_t start_offload, end_offload;
    fprintf(stderr, "Offloading GPU layers ...");
    time_measure(start_offload);
    rwkv_gpu_offload_layers(ctx, 16);
    time_measure(end_offload);
    fprintf(stderr, " %.3fs\n", TIME_DIFF(freq, start_offload, end_offload));

    const uint32_t tokens[128] = { 0 };
    const uint32_t num_tokens = sizeof(tokens) / sizeof(*tokens);

    size_t state_nelems = rwkv_get_state_buffer_element_count(ctx);
    size_t logits_nelems = rwkv_get_logits_buffer_element_count(ctx);

    float * state = calloc(state_nelems, sizeof(float));
    float * state_seq2 = calloc(state_nelems, sizeof(float));
    float * state_seq1 = calloc(state_nelems, sizeof(float));
    float * logits = calloc(logits_nelems, sizeof(float));
    float * logits_seq1 = calloc(logits_nelems, sizeof(float));
    float * logits_seq2 = calloc(logits_nelems, sizeof(float));

    time_t start_serial, end_serial;
    fprintf(stderr, "Serial mode to process %" PRId32 " tokens ...", num_tokens);
    time_measure(start_serial);
    for (uint32_t i = 0; i < num_tokens; i++)
        rwkv_eval(ctx, tokens[i], i == 0 ? NULL : state, state, logits);
    time_measure(end_serial);
    double serial_diff = TIME_DIFF(freq, start_serial, end_serial);
    fprintf(stderr, " %.3fs\n", serial_diff);

    time_t start_seq, end_seq;
    fprintf(stderr, "Sequence mode to process %" PRId32 " tokens ...", num_tokens);
    time_measure(start_seq);
    rwkv_eval_sequence(ctx, tokens, num_tokens, NULL, state_seq1, logits_seq1);
    time_measure(end_seq);
    double sequence_build_diff = TIME_DIFF(freq, start_seq, end_seq);
    fprintf(stderr, " 1 = %.3fs ...", sequence_build_diff);
    time_measure(start_seq);
    rwkv_eval_sequence(ctx, tokens, num_tokens, NULL, state_seq2, logits_seq2);
    time_measure(end_seq);
    double sequence_diff = TIME_DIFF(freq, start_seq, end_seq);
    fprintf(stderr, " 2 = %.3fs\n", sequence_diff);

    fprintf(stderr, "Factor: 1 = %.3fx, 2 = %.3fx\n", serial_diff / sequence_build_diff, serial_diff / sequence_diff);

    fprintf(stderr, "\n");
    fprintf(stderr, "State1 identical = %s\n", memcmp(state, state_seq1, state_nelems * sizeof(float)) == 0 ? "TRUE" : "FALSE");
    fprintf(stderr, "State2 identical = %s\n", memcmp(state, state_seq2, state_nelems * sizeof(float)) == 0 ? "TRUE" : "FALSE");
    fprintf(stderr, "Logits1 identical = %s\n", memcmp(logits, logits_seq1, logits_nelems * sizeof(float)) == 0 ? "TRUE" : "FALSE");
    fprintf(stderr, "Logits2 identical = %s\n", memcmp(logits, logits_seq2, logits_nelems * sizeof(float)) == 0 ? "TRUE" : "FALSE");

    float logits1_diff = 0;
    float logits2_diff = 0;

    for (size_t i = 0; i < logits_nelems; i++) {
        logits1_diff += logits[i] - logits_seq1[i];
        logits2_diff += logits[i] - logits_seq2[i];
    }

    fprintf(stderr, "Logits1 total diff = %.5f\n", logits1_diff);
    fprintf(stderr, "Logits2 total diff = %.5f\n", logits2_diff);

    fprintf(stderr, "\n");
    fprintf(stderr, "Serial:     [");

    for (size_t i = 0; i < 5; i++) {
        fprintf(stderr, " %.5f", logits[i]);
    }

    fprintf(stderr, " ...");

    for (size_t i = logits_nelems - 5; i < logits_nelems; i++) {
        fprintf(stderr, " %.5f", logits[i]);
    }

    fprintf(stderr, " ]\n");

    fprintf(stderr, "Sequence 1: [");

    for (size_t i = 0; i < 5; i++) {
        fprintf(stderr, " %.5f", logits_seq1[i]);
    }

    fprintf(stderr, " ...");

    for (size_t i = logits_nelems - 5; i < logits_nelems; i++) {
        fprintf(stderr, " %.5f", logits_seq1[i]);
    }

    fprintf(stderr, " ]\n");

    fprintf(stderr, "Sequence 2: [");

    for (size_t i = 0; i < 5; i++) {
        fprintf(stderr, " %.5f", logits_seq2[i]);
    }

    fprintf(stderr, " ...");

    for (size_t i = logits_nelems - 5; i < logits_nelems; i++) {
        fprintf(stderr, " %.5f", logits_seq2[i]);
    }

    fprintf(stderr, " ]\n");

    rwkv_free(ctx);

    return EXIT_SUCCESS;
}