static void rwkv_validate_tensors_for_custom_unary_op(struct ggml_tensor * dest, const struct ggml_tensor * src) {
    GGML_ASSERT(dest->type == GGML_TYPE_F32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dest));
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_are_same_shape(src, dest));
    // Verify that the shape is 2D.
    GGML_ASSERT(dest->ne[2] == 1);
    GGML_ASSERT(dest->ne[3] == 1);
}

#define SUPPRESS_UNUSED_WARNINGS_IN_CUSTOM_OP() { (void) ith; (void) nth; (void) userdata; }

static void rwkv_exp_impl(struct ggml_tensor * dest, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    rwkv_validate_tensors_for_custom_unary_op(dest, src);

    int64_t element_count = src->ne[0] * src->ne[1];
    int64_t start = ith * element_count / nth;
    int64_t end = (ith + 1) * element_count / nth;
    float * src_data = (float *) src->data;
    float * dest_data = (float *) dest->data;

    for (int64_t i = start; i < end; i++) {
        dest_data[i] = expf(src_data[i]);
    }

    SUPPRESS_UNUSED_WARNINGS_IN_CUSTOM_OP();
}

static void rwkv_1_minus_x_impl(struct ggml_tensor * dest, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    rwkv_validate_tensors_for_custom_unary_op(dest, src);

    int64_t element_count = src->ne[0] * src->ne[1];
    int64_t start = ith * element_count / nth;
    int64_t end = (ith + 1) * element_count / nth;
    float * src_data = (float *) src->data;
    float * dest_data = (float *) dest->data;

    for (int64_t i = start; i < end; i++) {
        dest_data[i] = 1.0F - src_data[i];
    }

    SUPPRESS_UNUSED_WARNINGS_IN_CUSTOM_OP();
}

static void rwkv_sigmoid_impl(struct ggml_tensor * dest, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    rwkv_validate_tensors_for_custom_unary_op(dest, src);

    int64_t element_count = src->ne[0] * src->ne[1];
    int64_t start = ith * element_count / nth;
    int64_t end = (ith + 1) * element_count / nth;
    float * src_data = (float *) src->data;
    float * dest_data = (float *) dest->data;

    for (int64_t i = start; i < end; i++) {
        dest_data[i] = 1.0F / (1.0F + expf(-src_data[i]));
    }

    SUPPRESS_UNUSED_WARNINGS_IN_CUSTOM_OP();
}

static void rwkv_max_impl(
    struct ggml_tensor * dest,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    int ith,
    int nth,
    void * userdata
) {
    GGML_ASSERT(dest->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dest));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_are_same_shape(src0, dest));
    GGML_ASSERT(ggml_are_same_shape(src1, dest));
    // Verify that the shape is 2D.
    GGML_ASSERT(dest->ne[2] == 1);
    GGML_ASSERT(dest->ne[3] == 1);

    int64_t element_count = src0->ne[0] * src0->ne[1];
    int64_t start = ith * element_count / nth;
    int64_t end = (ith + 1) * element_count / nth;
    float * src0_data = (float *) src0->data;
    float * src1_data = (float *) src1->data;
    float * dest_data = (float *) dest->data;

    for (int64_t i = start; i < end; i++) {
        dest_data[i] = fmaxf(src0_data[i], src1_data[i]);
    }

    SUPPRESS_UNUSED_WARNINGS_IN_CUSTOM_OP();
}

// From ggml.c
static void rwkv_groupnorm_impl(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    int ith,
    int nth,
    void * userdata
) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS

    const float eps = ((float*)userdata)[0];
    const int n_groups = ((int32_t*)userdata)[1];

    int n_channels = src0->ne[2];
    int n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
    for (int i = ith; i < n_groups; i += nth) {
        int start = i * n_channels_per_group;
        int end = start + n_channels_per_group;
        if (end > n_channels) {
            end = n_channels;
        }
        int step = end - start;

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            float sum = 0.0;
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * x = (float *)((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    float sumr = 0.0;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (float)x[i00];
                    }
                    sum += sumr;
                }
            }
            const float mean = sum / (ne00 * ne01 * step);

            float sum2 = 0.0;
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * x = (float *)((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    float * y = (float *)((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    float sumr = 0.0;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        float v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            const float variance = sum2 / (ne00 * ne01 * step);
            const float scale = 1.0f / sqrtf(variance + eps);

            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    float * y = (float *)((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    for (int i00 = 0; i00 < ne00; i00++) {
                        y[i00] *= scale;
                    }
                }
            }
        }
    }

    SUPPRESS_UNUSED_WARNINGS_IN_CUSTOM_OP();
}

// Element-wise exp(x)
struct ggml_tensor * rwkv_exp(struct ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_map_custom1(ctx, x, rwkv_exp_impl, 1, NULL);
}

// Element-wise 1 - x
struct ggml_tensor * rwkv_1_minus_x(struct ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_map_custom1(ctx, x, rwkv_1_minus_x_impl, 1, NULL);
}

// Element-wise sigmoid(x)
struct ggml_tensor * rwkv_sigmoid_inplace(struct ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_map_custom1_inplace(ctx, x, rwkv_sigmoid_impl, 1, NULL);
}

// Element-wise max(x, y)
struct ggml_tensor * rwkv_max(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y) {
    return ggml_map_custom2(ctx, x, y, rwkv_max_impl, 1, NULL);
}

// GroupNorm with custom eps value; Remove when ggml_norm supports eps as an argument.
struct ggml_tensor * rwkv_group_norm_eps_1e_minus5_inplace(struct ggml_context * ctx, struct ggml_tensor * x, int n_groups) {
    static float params[2];
    params[0] = 1e-5F;
    ((int*)params)[1] = n_groups;
    return ggml_map_custom1_inplace(ctx, x, rwkv_groupnorm_impl, 1, params);
}

struct ggml_tensor * rwkv_group_norm_eps_64e_minus5_inplace(struct ggml_context * ctx, struct ggml_tensor * x, int n_groups) {
    static float params[2];
    params[0] = 64e-5F;
    ((int*)params)[1] = n_groups;
    return ggml_map_custom1_inplace(ctx, x, rwkv_groupnorm_impl, 1, params);
}

struct ggml_tensor * rwkv_layer_norm(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * weight, struct ggml_tensor * bias) {
    // LayerNorm in RWKV is `x = (x - mean(x)) / sqrt(variance(x) + 1e-5) * weight + bias`
    // Looks like ggml_norm does the first part, we only need to apply weight & bias.
    return ggml_add_inplace(ctx, ggml_mul_inplace(ctx, ggml_norm(ctx, x, 1e-5F), weight), bias);
}
