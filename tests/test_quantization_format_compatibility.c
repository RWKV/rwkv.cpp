// Tests that existing Q5_0 & Q5_1 model files are still working.

#include "ggml.h"
#include "rwkv.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "logit_difference_validator.inc"

int main(void) {
    fprintf(stderr, "System info: %s\n", rwkv_get_system_info_string());

    float * expected_logits = malloc(sizeof(float) * N_VOCAB);
    load_expected_logits(expected_logits);

    test_model("tiny-rwkv-660K-Q5_0.bin", expected_logits, -0.170404F);
    test_model("tiny-rwkv-660K-Q5_1.bin", expected_logits, 0.278034F);

    free(expected_logits);

    return 0;
}
