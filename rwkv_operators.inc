void rwkv_exp_impl(struct ggml_tensor * dest, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    GGML_ASSERT(dest->type == GGML_TYPE_F32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dest));
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_are_same_shape(src, dest));

    // Assuming 2D tensors.
    int64_t element_count = src->ne[0] * src->ne[1];
    float * src_data = (float *) src->data;
    float * dest_data = (float *) dest->data;

    for (int64_t i = 0; i < element_count; i++) {
        dest_data[i] = expf(src_data[i]);
    }

    // Suppress warnings for unused parameters.
    (void) ith;
    (void) nth;
    (void) userdata;
}

void rwkv_1_minus_x_impl(struct ggml_tensor * dest, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    GGML_ASSERT(dest->type == GGML_TYPE_F32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dest));
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_are_same_shape(src, dest));

    // Assuming 2D tensors.
    int64_t element_count = src->ne[0] * src->ne[1];
    float * src_data = (float *) src->data;
    float * dest_data = (float *) dest->data;

    for (int64_t i = 0; i < element_count; i++) {
        dest_data[i] = 1.0F - src_data[i];
    }

    // Suppress warnings for unused parameters.
    (void) ith;
    (void) nth;
    (void) userdata;
}

void rwkv_sigmoid_impl(struct ggml_tensor * dest, const struct ggml_tensor * src, int ith, int nth, void * userdata) {
    GGML_ASSERT(dest->type == GGML_TYPE_F32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dest));
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_are_same_shape(src, dest));

    // Assuming 2D tensors.
    int64_t element_count = src->ne[0] * src->ne[1];
    float * src_data = (float *) src->data;
    float * dest_data = (float *) dest->data;

    for (int64_t i = 0; i < element_count; i++) {
        dest_data[i] = 1.0F / (1.0F + expf(-src_data[i]));
    }

    // Suppress warnings for unused parameters.
    (void) ith;
    (void) nth;
    (void) userdata;
}

void rwkv_max_impl(
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

    // Assuming 2D tensors.
    int64_t element_count = src0->ne[0] * src0->ne[1];
    float * src0_data = (float *) src0->data;
    float * src1_data = (float *) src1->data;
    float * dest_data = (float *) dest->data;

    for (int64_t i = 0; i < element_count; i++) {
        dest_data[i] = fmaxf(src0_data[i], src1_data[i]);
    }

    // Suppress warnings for unused parameters.
    (void) ith;
    (void) nth;
    (void) userdata;
}

struct ggml_tensor * rwkv_exp(ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_map_custom1(ctx, x, rwkv_exp_impl, 1, NULL);
}

struct ggml_tensor * rwkv_1_minus_x(ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_map_custom1(ctx, x, rwkv_1_minus_x_impl, 1, NULL);
}

struct ggml_tensor * rwkv_sigmoid(ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_map_custom1(ctx, x, rwkv_sigmoid_impl, 1, NULL);
}

struct ggml_tensor * rwkv_max(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y) {
    return ggml_map_custom2(ctx, x, y, rwkv_max_impl, 1, NULL);
}

struct ggml_tensor * rwkv_layer_norm(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * weight, struct ggml_tensor * bias) {
    // LayerNorm in RWKV is `x = (x - mean(x)) / sqrt(variance(x) + 1e-5) * weight + bias`
    // Looks like ggml_norm does the first part, we only need to apply weight & bias.
    return ggml_add_inplace(ctx, ggml_mul_inplace(ctx, ggml_norm(ctx, x, 1e-5F), weight), bias);
}
