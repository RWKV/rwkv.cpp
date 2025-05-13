// Ported from https://github.com/harrisonvanderbyl/RNN-Factory/blob/3b696b547cc9e25de04a077602c3fe1133d8984c/src/models/modules/cuda/cpuonly.cpp#L8
// Original code by Harrison Vanderbyl.
// TODO Fix 1. unaligned memory access on Linux with AVX2, 2. tiny-rwkv with AVX-512
#ifdef __AVX512F__
    #include <immintrin.h>
    #define SIMD_WIDTH       16
    #define LOAD(x)          _mm512_load_ps(x)
    #define STORE(x, y)      _mm512_store_ps(x, y)
    #define SET1(x)          _mm512_set1_ps(x)
    #define MULTIPLY(x, y)   _mm512_mul_ps(x, y)
    #define MULTADD(x, y, z) _mm512_fmadd_ps(x, y, z)
    #define ADD(x, y)        _mm512_add_ps(x, y)
    #define ZEROS()           _mm512_setzero_ps()
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define SIMD_WIDTH       8
    #define LOAD(x)          _mm256_load_ps(x)
    #define STORE(x, y)      _mm256_store_ps(x, y)
    #define SET1(x)          _mm256_set1_ps(x)
    #define MULTIPLY(x, y)   _mm256_mul_ps(x, y)
    #define MULTADD(x, y, z) _mm256_fmadd_ps(x, y, z)
    #define ADD(x, y)        _mm256_add_ps(x, y)
    #define ZEROS()           _mm256_setzero_ps()
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define SIMD_WIDTH       4
    #define LOAD(x)          vld1q_f32(x)
    #define STORE(x, y)      vst1q_f32(x, y)
    #define SET1(x)          vdupq_n_f32(x)
    #define MULTIPLY(x, y)   vmulq_f32(x, y)
    #define MULTADD(x, y, z) vmlaq_f32(z, x, y)
#else
    #define SIMD_WIDTH       1
    #define LOAD(x)          *x
    #define STORE(x, y)      *x = y
    #define SET1(x)          x
    #define MULTIPLY(x, y)   x * y
    #define MULTADD(x, y, z) x * y + z
#endif


inline float horizontal_sum(__m256 vec) {
    __m256 sum1 = _mm256_hadd_ps(vec, vec);
    __m256 sum2 = _mm256_hadd_ps(sum1, sum1);
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum2, 0), 
                              _mm256_extractf128_ps(sum2, 1));
    float result;
    _mm_store_ss(&result, sum128);
    return result;
}

static void rwkv_wkv_v7_impl(struct ggml_tensor * result, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    // const size_t T = result->ne[1];
    const size_t C = result->ne[0];
    const size_t S = result->src[1]->ne[0];
    const size_t H = result->src[1]->ne[1];
    const size_t T = result->src[1]->ne[2];
    GGML_ASSERT(C == S * H);
    
    float * result_data = (float *) result->data;
    float * state_out = (float *) result->data + C * T;

    float * state = (float *) src->data;
    float * r =     (float *) result->src[1]->data;
    float * w =     (float *) result->src[2]->data;
    float * k =     (float *) result->src[3]->data;
    float * v =     (float *) result->src[4]->data;
    float * a =     (float *) result->src[5]->data;
    float * b =     (float *) result->src[6]->data;

    size_t t_stride = H * S;

    size_t h_stride = C / H;
    size_t h_stride_2d = S * S;

    for (size_t t = 0; t < T; t++) {
        size_t t_offset = t * t_stride;

        float * state_in = (t == 0) ? state : state_out;
        for (size_t h = ith; h < H; h += nth) {
            size_t h_offset = h * h_stride;
            size_t t_h_offset = t_offset + h_offset;
            size_t h_2d_offset = h * h_stride_2d;

            for (size_t i = 0; i < C / H; i ++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;
                    if (i == 0) {
                        memset(&result_data[t_h_offset], 0, h_stride * sizeof(float));
                    }

                    auto sa_vec = ZEROS();
                    for (size_t j = 0; j < C / H; j += SIMD_WIDTH) {
                        sa_vec = ADD(sa_vec, MULTIPLY(
                                                LOAD(&a[t_h_offset + j]), 
                                                LOAD(&state_in[h_2d_i_offset + j])
                                            )
                                    );
                    }
                    float sa = horizontal_sum(sa_vec);

                    auto v_vec = SET1(v[t_h_i_offset]);
                    sa_vec = SET1(sa);

                    auto sum = ZEROS();
                    for (size_t j = 0; j < C / H; j += SIMD_WIDTH) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        auto r_val = LOAD(&r[t_h_j_offset]);
                        auto w_val = LOAD(&w[t_h_j_offset]);
                        auto k_val = LOAD(&k[t_h_j_offset]);
                        auto b_val = LOAD(&b[t_h_j_offset]);
                        auto prev_state_val = LOAD(&state_in[h_2d_i_j_offset]);

                        // auto kv_val = v_val * k_val;
                        auto kv_val = MULTIPLY(v_vec, k_val);

                        // state_out[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                        auto sab_val = MULTIPLY(sa_vec, b_val);
                        auto state_out_val = MULTADD(prev_state_val, w_val, kv_val);
                        state_out_val = ADD(state_out_val, sab_val);
                        STORE(&state_out[h_2d_i_j_offset], state_out_val);

                        // result_data[t_h_i_offset] += state_out[h_2d_i_j_offset] * r_val;
                        auto result = MULTIPLY(state_out_val, r_val);

                        // auto sum = LOAD(&result_data[t_h_i_offset]);
                        sum = ADD(sum, result);
                    }
                    result_data[t_h_i_offset] = horizontal_sum(sum);
            }
            
        }
    }

    // Suppress "unused parameter" warnings.
    (void) src;
    (void) nth;
    (void) userdata;
}

// Parameters:
// - T: sequence length
// - C: channel count, same as n_embed
// - H: head count
// - S: head size
// Shapes (in ggml order):
// - r:          [S, H, T]
// - w:          [S, H, T]
// - k:          [S, H, T]
// - v:          [S, H, T]
// - a:          [S, H, T]
// - b:          [S, H, T]
// - state:      [S * S * H, 1, 1, 1]
// - result:     concated output + state_output
static struct ggml_tensor * rwkv_wkv_v7(
    struct ggml_context * ctx,
    struct ggml_tensor * state,
    struct ggml_tensor * r,
    struct ggml_tensor * w,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    struct ggml_tensor * a,
    struct ggml_tensor * b
) {
    GGML_ASSERT(r->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32);
    GGML_ASSERT(v->type == GGML_TYPE_F32);
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);
    GGML_ASSERT(state->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    const int64_t S = r->ne[0];
    const int64_t H = r->ne[1];
    const int64_t T = r->ne[2];
    const int64_t C = S * H;

    GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == T);
    GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == T);
    GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == T);
    GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == T);
    GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == T);
    GGML_ASSERT(ggml_nelements(state) == S * S * H);

    struct ggml_tensor * result = ggml_map_custom1(
        ctx,
        state,
        rwkv_wkv_v7_impl,
        1,
        NULL
    );
    result->src[1] = r;
    result->src[2] = w;
    result->src[3] = k;
    result->src[4] = v;
    result->src[5] = a;
    result->src[6] = b;

    result->ne[0] = C;
    result->ne[1] = T + S;

    return result;
}
