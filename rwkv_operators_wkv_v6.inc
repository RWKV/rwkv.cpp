#include "rwkv_operators_wkv_common.inc"

// Ported from https://github.com/harrisonvanderbyl/RNN-Factory/blob/3b696b547cc9e25de04a077602c3fe1133d8984c/src/models/modules/cuda/cpuonly.cpp#L57
// Original code by Harrison Vanderbyl.
static void rwkv_wkv_v6_impl(struct ggml_tensor * result, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    const size_t T = result->ne[1];
    const size_t C = result->ne[0];
    const size_t H = result->src[1]->ne[2];

    // TODO: Multi-threading.
    if (ith != 0)
        return;

    float * result_data = (float *) result->data;

    memset(result_data, 0, T * C * sizeof(float));

    float * k =          (float *) result->src[1]->data;
    float * v =          (float *) result->src[2]->data;
    float * r =          (float *) result->src[3]->data;
    float * time_faaaa = (float *) result->src[4]->data;
    float * time_decay = (float *) result->src[5]->data;
    float * state =      (float *) result->src[6]->data;

    size_t t_stride = H * (C / H);

    size_t h_stride = C / H;
    size_t h_stride_2d = (C / H) * (C / H);

    for (size_t t = 0; t < T; t++) {
        size_t t_offset = t * t_stride;

        for (size_t h = 0; h < H; h++) {
            size_t h_offset = h * h_stride;
            size_t t_h_offset = t_offset + h_offset;
            size_t h_2d_offset = h * h_stride_2d;

            for (size_t i = 0; i < C / H; i++) {
                size_t t_h_i_offset = t_h_offset + i;
                size_t h_i_offset = h_offset + i;
                size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                auto k_val = SET1(k[t_h_i_offset]);
                auto r_val = SET1(r[t_h_i_offset]);
                auto time_faaaa_val = SET1(time_faaaa[h_i_offset]);
                // RWKV v6: different time_decay for each token.
                auto time_decay_val = SET1(time_decay[t_h_i_offset]);

                for (size_t j = 0; j < C / H; j += SIMD_WIDTH) {
                    size_t t_h_j_offset = t_h_offset + j;
                    size_t h_2d_i_j_offset = h_2d_i_offset + j;

                    auto v_val = LOAD(&v[t_h_j_offset]);

                    auto kv_val = MULTIPLY(v_val, k_val);

                    auto prev_state_val = LOAD(&state[h_2d_i_j_offset]);

                    auto temp_val = MULTADD(kv_val, time_faaaa_val, prev_state_val);

                    auto prev_result_data = LOAD(&result_data[t_h_j_offset]);

                    STORE(&result_data[t_h_j_offset], MULTADD(temp_val, r_val, prev_result_data));

                    STORE(&state[h_2d_i_j_offset], MULTADD(prev_state_val, time_decay_val, kv_val));
                }
            }
        }
    }

    // Suppress "unused parameter" warnings.
    (void) src;
    (void) ith;
    (void) nth;
    (void) userdata;
}

// Parameters:
// - T: sequence length
// - C: channel count, same as n_embed
// - H: head count
// - S: head size
// Shapes (in ggml order):
// - x:          [C, T, 1, 1]
// - k:          [1, S, H, T]
// - v:          [S, 1, H, T]
// - r:          [S, 1, H, T]
// - time_faaaa: [1, S, H, 1]
// - w:          [1, S, H, T]
// - state:      [S * S * H, 1, 1, 1]
// - result:     same as x
// state will be written to.
static struct ggml_tensor * rwkv_wkv_v6(
    struct ggml_context * ctx,
    const size_t T,
    const size_t C,
    const size_t H,
    const size_t S,
    struct ggml_tensor * x,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    struct ggml_tensor * r,
    struct ggml_tensor * time_faaaa,
    struct ggml_tensor * w,
    struct ggml_tensor * state
) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32);
    GGML_ASSERT(v->type == GGML_TYPE_F32);
    GGML_ASSERT(r->type == GGML_TYPE_F32);
    GGML_ASSERT(time_faaaa->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(state->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(time_faaaa));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(state));

    GGML_ASSERT(x->ne[0] == C && x->ne[1] == T && x->ne[2] == 1 && x->ne[3] == 1);
    GGML_ASSERT(k->ne[0] == 1 && k->ne[1] == S && k->ne[2] == H && k->ne[3] == T);
    GGML_ASSERT(v->ne[0] == S && v->ne[1] == 1 && v->ne[2] == H && v->ne[3] == T);
    GGML_ASSERT(r->ne[0] == S && r->ne[1] == 1 && r->ne[2] == H && r->ne[3] == T);
    GGML_ASSERT(w->ne[0] == 1 && w->ne[1] == S && w->ne[2] == H && w->ne[3] == T);
    GGML_ASSERT(ggml_nelements(state) == S * S * H);

    // k = ggml_cont_4d(ctx, ggml_transpose(ctx, k), S, 1, H, T);
    // v = ggml_cont_4d(ctx, ggml_transpose(ctx, v), 1, S, H, T);
    // r = ggml_cont_4d(ctx, ggml_transpose(ctx, r), 1, S, H, T);
    k = ggml_transpose(ctx, k);
    v = ggml_transpose(ctx, v);
    r = ggml_transpose(ctx, r);

    struct ggml_tensor * result = ggml_map_custom1(
        ctx,
        x,
        rwkv_wkv_v6_impl,
        1,
        NULL
    );
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = r;
    result->src[4] = time_faaaa;
    result->src[5] = w;
    result->src[6] = state;

    return result;
}
