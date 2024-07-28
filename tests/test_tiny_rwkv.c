// Tests that tiny RWKV outputs expected results in all data types.
#include <stdlib.h>
#include <stdio.h>

#include <rwkv.h>

#include "logit_difference_validator.inc"

#define VERSION_COUNT 4
#define FORMAT_COUNT 7

int main(void) {
    late_abort = true;

    fprintf(stderr, "System info: %s\n", rwkv_get_system_info_string());

    // Silences the overly verbose output during quantization.
    rwkv_set_print_errors(NULL, false);

    const char * versions[VERSION_COUNT] = {
        "4v0-660K",
        "5v1-730K",
        "5v2-730K",
        "6v0-3m"
    };

    const char * formats[FORMAT_COUNT] = {
        "FP32",
        "FP16",
        "Q4_0",
        "Q4_1",
        "Q5_0",
        "Q5_1",
        "Q8_0"
    };

    const float expected_difference_sum_full[VERSION_COUNT * 2] = {
        // 4v0
        +0.001000F, // FP32
        -0.013652F, // FP16
        // 5v1
        +0.001000F, // FP32
        -0.289921F, // FP16
        // 5v2
        +0.001000F, // FP32
        +0.455912F, // FP16
        // 6v0
        +0.001000F, // FP32
        -0.416620F  // FP16
    };

    // *** Why the hell the expected logit difference sum for v4 models is < 1, and for v5 models it can be as high as 160? ***
    //
    // Due to mistake in Tiny RWKV v4 training code, all FFN layers were zeroed-out during training.
    // Output of v4 models is basically incoherent -- there are "words" and spaces between them, but these words consist of random characters.
    // I quess that since there is not much "intelligence" to lose, there will be a pretty low logit difference sum after quantization.
    //
    // In contrast, Tiny RWKV v5 models were trained correctly, and FFN layers have an OK-looking weight distribution.
    // v5 models produce mostly real English words, and, sometimes, whole word combinations that make sense. Structure of the output is also correct.
    // Since there are real numbers in FFN layers now, I expect quantization to have a way larger effect on the output, compared to v4.
    //
    // For reference, RWKV v4 169M would give -2395.1636 logit difference sum after quantizing FP32 to Q5_1. So, such orders of magnitude are not unheard of.
    //
    // In any case, here, the logit difference sum works OK for verifying that inference was not broken after some changes.

    const float expected_difference_sum_quantized_FP32[VERSION_COUNT * (FORMAT_COUNT - 2)] = {
         // 4v0
        -000.160030F, // Q4_0
        -000.547409F, // Q4_1
        -000.170404F, // Q5_0
        +000.278034F, // Q5_1
        +000.076282F, // Q8_0
        // 5v1
        +117.932594F, // Q4_0
        -026.712271F, // Q4_1
        -163.439407F, // Q5_0
        -018.017435F, // Q5_1
        +000.585238F, // Q8_0
        // 5v2
        +035.271305F, // Q4_0
        +067.015076F, // Q4_1
        +025.273308F, // Q5_0
        +048.068733F, // Q5_1
        -009.441034F, // Q8_0
        // 6v0
        -007.588121F, // Q4_0
        +021.939022F, // Q4_1
        -027.332073F, // Q5_0
        +003.576909F, // Q5_1
        -009.539596F  // Q8_0
    };

    const float expected_difference_sum_quantized_FP16[VERSION_COUNT * (FORMAT_COUNT - 2)] = {
         // 4v0
        +000.154614F, // Q4_0
        -000.539827F, // Q4_1
        -000.180142F, // Q5_0
        +000.294953F, // Q5_1
        +000.077226F, // Q8_0
        // 5v1
        +119.471931F, // Q4_0
        -028.245888F, // Q4_1
        -159.870956F, // Q5_0
        -039.708530F, // Q5_1
        -000.962695F, // Q8_0
        // 5v2
        +034.135971F, // Q4_0
        +065.573822F, // Q4_1
        +021.588751F, // Q5_0
        +029.726818F, // Q5_1
        -007.242277F, // Q8_0
        // 6v0
        -007.660988F, // Q4_0
        +021.797060F, // Q4_1
        -027.269241F, // Q5_0
        +003.405264F, // Q5_1
        -009.734720F  // Q8_0
    };

    for (int i_version = 0; i_version < VERSION_COUNT; i_version++) {
        float * expected_logits = calloc(N_VOCAB, sizeof(float));
        load_expected_logits(expected_logits, versions[i_version]);

        for (int i_format = 0; i_format < FORMAT_COUNT; i_format++) {
            if (i_format < 2) {
                test_model(versions[i_version], formats[i_format], expected_logits, expected_difference_sum_full[i_version * 2 + i_format]);

                continue;
            }

            char source_file_name[128];
            char dest_format[128];
            char dest_file_name[128];

            // ---

            sprintf(source_file_name, "tiny-rwkv-%s-FP32.bin", versions[i_version]);
            sprintf(dest_format, "FP32-to-%s", formats[i_format]);
            sprintf(dest_file_name, "tiny-rwkv-%s-%s.bin", versions[i_version], dest_format);

            rwkv_quantize_model_file(source_file_name, dest_file_name, formats[i_format]);

            test_model(versions[i_version], dest_format, expected_logits, expected_difference_sum_quantized_FP32[i_version * (FORMAT_COUNT - 2) + (i_format - 2)]);

            // ---

            sprintf(source_file_name, "tiny-rwkv-%s-FP16.bin", versions[i_version]);
            sprintf(dest_format, "FP16-to-%s", formats[i_format]);
            sprintf(dest_file_name, "tiny-rwkv-%s-%s.bin", versions[i_version], dest_format);

            rwkv_quantize_model_file(source_file_name, dest_file_name, formats[i_format]);

            test_model(versions[i_version], dest_format, expected_logits, expected_difference_sum_quantized_FP16[i_version * (FORMAT_COUNT - 2) + (i_format - 2)]);
        }

        free(expected_logits);
    }

    if (must_abort) {
        abort();
    }

    return 0;
}
