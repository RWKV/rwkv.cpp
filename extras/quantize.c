#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <ggml.h>
#include <rwkv.h>

#if defined(_WIN32)
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

static enum ggml_type type_from_string(const char * string) {
    if (strcmp(string, "Q4_0") == 0) return GGML_TYPE_Q4_0;
    if (strcmp(string, "Q4_1") == 0) return GGML_TYPE_Q4_1;
    if (strcmp(string, "Q4_K") == 0) return GGML_TYPE_Q4_K;
    if (strcmp(string, "Q5_0") == 0) return GGML_TYPE_Q5_0;
    if (strcmp(string, "Q5_1") == 0) return GGML_TYPE_Q5_1;
    if (strcmp(string, "Q5_K") == 0) return GGML_TYPE_Q5_K;
    if (strcmp(string, "Q8_0") == 0) return GGML_TYPE_Q8_0;
    return GGML_TYPE_COUNT;
}

int main(const int argc, const char * argv[]) {
    if (argc != 4 || type_from_string(argv[3]) == GGML_TYPE_COUNT) {
        fprintf(stderr, "Usage: %s INPUT_FILE OUTPUT_FILE FORMAT\n\nAvailable formats: Q4_0 Q4_1 Q5_0 Q5_1 Q8_0\n", argv[0]);

        return EXIT_FAILURE;
    }

    time_t freq, start, end;
    time_calibrate(freq);

    fprintf(stderr, "Quantizing...\n");

    time_measure(start);
    bool success = rwkv_quantize_model_file(argv[1], argv[2], argv[3]);
    time_measure(end);

    double diff = TIME_DIFF(freq, start, end);

    fprintf(stderr, "Took %.3f s\n", diff);

    if (success) {
        fprintf(stderr, "Success\n");

        return EXIT_SUCCESS;
    } else {
        fprintf(stderr, "Error: 0x%.8X\n", rwkv_get_last_error(NULL));

        return EXIT_FAILURE;
    }
}
