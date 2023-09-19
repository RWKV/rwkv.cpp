#ifndef ASSERTIONS_INC
#define ASSERTIONS_INC

#include <math.h>

#define ASSERT(x, ...) {\
        if (!(x)) {\
            fprintf(stderr, "*** Assertion failed ***\n");\
            fprintf(stderr, __VA_ARGS__);\
            fprintf(stderr, "\n%s:%d\n", __FILE__, __LINE__);\
            abort();\
        }\
    }

#define ASSERT_ELEMENT_F32(tensor, i, expected_value) {\
        float actual = ((float *) tensor->data)[i];\
        ASSERT(\
            fabsf(actual - expected_value) <= 0.0000001F,\
            "At %s[%d]: expected %f, actual %f",\
            #tensor,\
            i,\
            (double) expected_value,\
            (double) actual\
        );\
    }

#endif
