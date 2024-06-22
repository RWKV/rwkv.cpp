// Ported from https://github.com/harrisonvanderbyl/RNN-Factory/blob/3b696b547cc9e25de04a077602c3fe1133d8984c/src/models/modules/cuda/cpuonly.cpp#L8
// Original code by Harrison Vanderbyl.
// TODO Fix 1. unaligned memory access on Linux with AVX2, 2. tiny-rwkv with AVX-512
/*#ifdef __AVX512F__
    #include <immintrin.h>
    #define SIMD_WIDTH       16
    #define LOAD(x)          _mm512_load_ps(x)
    #define STORE(x, y)      _mm512_store_ps(x, y)
    #define SET1(x)          _mm512_set1_ps(x)
    #define MULTIPLY(x, y)   _mm512_mul_ps(x, y)
    #define MULTADD(x, y, z) _mm512_fmadd_ps(x, y, z)
#elif __AVX2__
    #include <immintrin.h>
    #define SIMD_WIDTH       8
    #define LOAD(x)          _mm256_load_ps(x)
    #define STORE(x, y)      _mm256_store_ps(x, y)
    #define SET1(x)          _mm256_set1_ps(x)
    #define MULTIPLY(x, y)   _mm256_mul_ps(x, y)
    #define MULTADD(x, y, z) _mm256_fmadd_ps(x, y, z)
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define SIMD_WIDTH       4
    #define LOAD(x)          vld1q_f32(x)
    #define STORE(x, y)      vst1q_f32(x, y)
    #define SET1(x)          vdupq_n_f32(x)
    #define MULTIPLY(x, y)   vmulq_f32(x, y)
    #define MULTADD(x, y, z) vmlaq_f32(z, x, y)
#else*/
    #define SIMD_WIDTH       1
    #define LOAD(x)          *x
    #define STORE(x, y)      *x = y
    #define SET1(x)          x
    #define MULTIPLY(x, y)   x * y
    #define MULTADD(x, y, z) x * y + z
//#endif
