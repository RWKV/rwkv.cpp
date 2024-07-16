// View tensors of a state of a single layer.
struct rwkv_layer_state {
    struct ggml_tensor * ffn_xx;
    struct ggml_tensor * att_xx;
    // Used in RWKV v4.
    struct ggml_tensor * att_aa;
    struct ggml_tensor * att_bb;
    struct ggml_tensor * att_pp;
    // Used in RWKV v5+.
    struct ggml_tensor * att_heads;
};

struct rwkv_ggml_cgraph_deleter {
    void operator()(struct ggml_cgraph * cgraph) {
        if (cgraph->nodes)
            free(cgraph->nodes);
        if (cgraph->leafs)
            free(cgraph->leafs);
        if (cgraph->visited_hash_table.keys)
            free(cgraph->visited_hash_table.keys);
        if (cgraph->grads)
            free(cgraph->grads);
        free(cgraph);
    }
};

static struct ggml_cgraph * rwkv_ggml_cgraph_create(size_t size, bool grads) {
    struct ggml_cgraph * cgraph = (struct ggml_cgraph *)calloc(1, sizeof(struct ggml_cgraph));
    cgraph->size = size;
    cgraph->n_nodes = 0;
    cgraph->n_leafs = 0;
    cgraph->nodes = (struct ggml_tensor **)calloc(1, size * sizeof(struct ggml_tensor *));
    cgraph->leafs = (struct ggml_tensor **)calloc(1, size * sizeof(struct ggml_tensor *));

    // next primes after powers of two
    static const size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    static const size_t n_primes = sizeof(primes)/sizeof(primes[0]);

    // find the smallest prime that is larger or equal to size
    size_t l = 0;
    size_t r = n_primes;
    while (l < r) {
        size_t m = (l + r)/2;
        if (primes[m] < size * 2) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    size_t hash_size = l < n_primes ? primes[l] : (size * 2 + 1);

    cgraph->visited_hash_table.size = hash_size;
    cgraph->visited_hash_table.keys = (struct ggml_tensor **)calloc(1, hash_size * sizeof(struct ggml_tensor *));
    cgraph->order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT;
    if (grads) {
        cgraph->grads = (struct ggml_tensor **)calloc(1, size * sizeof(struct ggml_tensor *));
    }
    return cgraph;
}


// The computation graph holds ggml context and the ggml cgraph.
// It can be either a serial or a sequential graph.
struct rwkv_computation_graph {
    struct ggml_context * ggml_ctx;
    // ggml_cgraph is so large that it can cause stack overflows if not stored on the heap.
    std::unique_ptr<struct ggml_cgraph, rwkv_ggml_cgraph_deleter> cgraph;
    ggml_backend_buffer_t buffer_inputs;

    // Input tensors.
    struct ggml_tensor * tokens;
    struct ggml_tensor * input_state;
    std::unique_ptr<struct rwkv_layer_state[]> input_layers;

    // Output tensors.
    struct ggml_tensor * output_state;
    std::unique_ptr<struct rwkv_layer_state[]> output_layers;
    struct ggml_tensor * logits;

    // ggml graph counters before the graph was extended with logits tensor.
    int pre_logits_nodes;
    int pre_logits_leafs;
    // ggml graph counters after the graph was extended with logits tensor.
    int post_logits_nodes;
    int post_logits_leafs;
};

// The context holds the model and both serial and sequential computation graphs.
struct rwkv_context {
    struct rwkv_model * model;

    // The serial graph implements the traditional RNN mode that processes only one token at a time (serial mode).
    struct rwkv_computation_graph serial_graph;
    // The sequence graph implements the "sequence mode" (or transformer/GPT mode) that processes multiple tokens at a time.
    // This can be an order of magnitude or so faster than serial execution if used properly.
    struct rwkv_computation_graph sequential_graph;
    size_t last_used_sequence_length;

    uint32_t n_threads;

    enum rwkv_error_flags last_error;
    bool print_errors;
};

static void rwkv_carry_x(
    struct ggml_context * ctx,
    struct ggml_tensor * weight,
    struct ggml_tensor * bias,
    struct ggml_tensor *& x,
    struct ggml_tensor *& x_prev,
    struct ggml_tensor *& carry
) {
    const size_t n_embed = x->ne[0];
    const size_t sequence_len = x->ne[1];

    if (sequence_len == 1) {
        // self.layer_norm(x, self.w.blocks[i].ln2)
        x = rwkv_layer_norm(ctx, x, weight, bias);

        // xx = state[5*i+0]
        x_prev = carry;

        // state[5*i+0] = x
        carry = x;
    } else {
        // self.layer_norm(x, self.w.blocks[i].ln2)
        x = rwkv_layer_norm(ctx, x, weight, bias);

        // xx = torch.cat((state[5*i+0].to(dtype=self.FLOAT_MODE).unsqueeze(0), x[:-1,:]))
        x_prev = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embed, sequence_len);
        x_prev = ggml_set_1d_inplace(ctx, x_prev, carry, 0);
        x_prev = ggml_set_1d_inplace(ctx, x_prev, ggml_view_1d(ctx, x, n_embed * (sequence_len - 1), 0), n_embed * sizeof(float));

        // state[5*i+0] = x[-1,:]
        carry = ggml_view_1d(ctx, x, n_embed, n_embed * (sequence_len - 1) * sizeof(float));
    }
}

static void rwkv_att_rkv(
    struct ggml_context * ctx,
    struct rwkv_layer layer,
    struct ggml_tensor * x,
    struct ggml_tensor * x_prev,
    struct ggml_tensor *& r,
    struct ggml_tensor *& k,
    struct ggml_tensor *& v
) {
    // xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
    struct ggml_tensor * xk = ggml_add_inplace(ctx,
        ggml_mul(ctx, x, layer.att_time_mix_k),
        ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.att_time_mix_k))
    );

    // xv = x * time_mix_v + state[5 * i + 1] * (1 - time_mix_v)
    struct ggml_tensor * xv = ggml_add_inplace(ctx,
        ggml_mul(ctx, x, layer.att_time_mix_v),
        ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.att_time_mix_v))
    );

    // xr = x * time_mix_r + state[5 * i + 1] * (1 - time_mix_r)
    struct ggml_tensor * xr = ggml_add_inplace(ctx,
        ggml_mul(ctx, x, layer.att_time_mix_r),
        ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.att_time_mix_r))
    );

    // r = torch.sigmoid(rw @ xr)
    r = rwkv_sigmoid_inplace(ctx, ggml_mul_mat(ctx, layer.att_receptance, xr));
    // k = kw @ xk
    k = ggml_mul_mat(ctx, layer.att_key, xk);
    // v = vw @ xv
    v = ggml_mul_mat(ctx, layer.att_value, xv);
}

static struct ggml_tensor * rwkv_att_wkv(
    struct ggml_context * ctx,
    struct ggml_tensor * att_time_first,
    struct ggml_tensor * att_time_decay,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    struct ggml_tensor *& aa,
    struct ggml_tensor *& bb,
    struct ggml_tensor *& pp
) {
    // ww = time_first + k
    struct ggml_tensor * ww = ggml_add(ctx, att_time_first, k);
    // qq = torch.maximum(pp, ww)
    struct ggml_tensor * qq = rwkv_max(ctx, pp, ww);
    // e1 = torch.exp(pp - qq)
    struct ggml_tensor * e1 = rwkv_exp(ctx, ggml_sub(ctx, pp, qq));
    // e2 = torch.exp(ww - qq)
    struct ggml_tensor * e2 = rwkv_exp(ctx, ggml_sub(ctx, ww, qq));

    // a = e1 * aa + e2 * v
    struct ggml_tensor * a = ggml_add_inplace(ctx, ggml_mul(ctx, e1, aa), ggml_mul(ctx, e2, v));
    // b = e1 * bb + e2
    struct ggml_tensor * b = ggml_add_inplace(ctx, ggml_mul(ctx, e1, bb), e2);

    // ww = pp + time_decay
    ww = ggml_add(ctx, pp, att_time_decay);
    // qq = torch.maximum(ww, k)
    qq = rwkv_max(ctx, ww, k);
    // e1 = torch.exp(ww - qq)
    e1 = rwkv_exp(ctx, ggml_sub(ctx, ww, qq));
    // e2 = torch.exp(k[t] - qq)
    e2 = rwkv_exp(ctx, ggml_sub(ctx, k, qq));

    // state[5 * i + 2] = e1 * aa + e2 * v
    // state[5 * i + 3] = e1 * bb + e2
    // state[5 * i + 4] = qq
    aa = ggml_add_inplace(ctx, ggml_mul(ctx, e1, aa), ggml_mul(ctx, e2, v));
    bb = ggml_add_inplace(ctx, ggml_mul(ctx, e1, bb), e2);
    pp = qq;

    // wkv = a / b
    return ggml_div(ctx, a, b);
}

static struct ggml_tensor * rwkv_att(struct ggml_context * ctx, struct ggml_tensor * x, struct rwkv_layer layer, struct rwkv_layer_state & state) {
    struct ggml_tensor * x_prev;
    rwkv_carry_x(ctx, layer.ln1_weight, layer.ln1_bias, x, x_prev, state.att_xx);

    struct ggml_tensor * r, * k, * v;
    rwkv_att_rkv(ctx, layer, x, x_prev, r, k, v);

    struct ggml_tensor * wkv = rwkv_att_wkv(ctx, layer.att_time_first, layer.att_time_decay, k, v, state.att_aa, state.att_bb, state.att_pp);

    // ow @ (r * xx)
    return ggml_mul_mat(ctx, layer.att_output, ggml_mul(ctx, r, wkv));
}

static struct ggml_tensor * rwkv_att_v5(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct rwkv_layer layer,
    struct rwkv_layer_state & state,
    const int64_t head_count,
    const int64_t head_size,
    const uint32_t arch_version_minor
) {
    size_t n_embed = x->ne[0];
    size_t sequence_length = x->ne[1];

    x = rwkv_layer_norm(ctx, x, layer.ln1_weight, layer.ln1_bias);

    struct ggml_tensor * x_prev;

    if (sequence_length > 1) {
        x_prev = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embed, sequence_length);
        x_prev = ggml_set_1d_inplace(ctx, x_prev, state.att_xx, 0);
        x_prev = ggml_set_1d_inplace(
            ctx,
            x_prev,
            ggml_view_1d(ctx, x, n_embed * (sequence_length - 1), 0), n_embed * sizeof(float)
        );
    } else {
        x_prev = state.att_xx;
    }

    struct ggml_tensor * xk = ggml_add_inplace(
        ctx,
        ggml_mul(ctx, x, layer.att_time_mix_k),
        ggml_mul(
            ctx,
            x_prev,
            rwkv_1_minus_x(ctx, layer.att_time_mix_k)
        )
    );

    struct ggml_tensor * xv = ggml_add_inplace(
        ctx,
        ggml_mul(ctx, x, layer.att_time_mix_v),
        ggml_mul(
            ctx,
            x_prev,
            rwkv_1_minus_x(ctx, layer.att_time_mix_v)
        )
    );

    struct ggml_tensor * xr = ggml_add_inplace(
        ctx,
        ggml_mul(ctx, x, layer.att_time_mix_r),
        ggml_mul(
            ctx,
            x_prev,
            rwkv_1_minus_x(ctx, layer.att_time_mix_r)
        )
    );

    struct ggml_tensor * xg = NULL;

    if (arch_version_minor >= 2) {
        xg = ggml_add_inplace(
            ctx,
            ggml_mul(ctx, x, layer.att_time_mix_g),
            ggml_mul(
                ctx,
                x_prev,
                rwkv_1_minus_x(ctx, layer.att_time_mix_g)
            )
        );
    }

    state.att_xx = ggml_view_1d(ctx, x, n_embed, n_embed * (sequence_length - 1) * sizeof(float));

    struct ggml_tensor * r = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, layer.att_receptance, xr), head_size, 1,         head_count, sequence_length);
    struct ggml_tensor * k = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, layer.att_key,        xk), 1,         head_size, head_count, sequence_length);
    struct ggml_tensor * v = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, layer.att_value,      xv), head_size, 1,         head_count, sequence_length);

    struct ggml_tensor * g = NULL;

    if (arch_version_minor >= 2) {
        g = ggml_silu_inplace(
            ctx,
            ggml_mul_mat(ctx, layer.att_gate, xg)
        );
    }

    // dup is not strictly required; doing it just in case.
    struct ggml_tensor * state_out = ggml_dup(ctx, state.att_heads);

    struct ggml_tensor * time_first;
    struct ggml_tensor * time_decay;

    if (arch_version_minor >= 2) {
        time_first = layer.att_time_faaaa;
        time_decay = layer.att_time_decay;
    } else {
        struct ggml_tensor * dummy = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, head_size, head_count);

        time_first = ggml_repeat(ctx, layer.att_time_first, dummy);
        time_decay = ggml_repeat(ctx, layer.att_time_decay, dummy);
    }

    x = rwkv_wkv_v5(
        ctx,
        sequence_length,
        n_embed,
        head_count,
        head_size,
        x,
        k,
        v,
        r,
        time_first,
        time_decay,
        state_out
    );

    state.att_heads = state_out;

    // ggml_group_norm considers groups in the third dimension.
    x = ggml_reshape_4d(ctx, x, 1, 1, n_embed, sequence_length);
    x = ggml_group_norm_inplace(ctx, x, head_count);
    // Convert back to a regular vector.
    x = ggml_reshape_2d(ctx, x, n_embed, sequence_length);
    x = ggml_add_inplace(
        ctx,
        ggml_mul_inplace(
            ctx,
            x,
            layer.att_ln_x_weight
        ),
        layer.att_ln_x_bias
    );

    if (arch_version_minor >= 2) {
        x = ggml_mul_inplace(ctx, x, g);
    }

    return ggml_mul_mat(ctx, layer.att_output, x);
}

static struct ggml_tensor * rwkv_att_v6(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct rwkv_layer layer,
    struct rwkv_layer_state & state,
    const int64_t head_count,
    const int64_t head_size,
    const uint32_t arch_version_minor
) {
    size_t n_embed = x->ne[0];
    size_t sequence_length = x->ne[1];

    x = rwkv_layer_norm(ctx, x, layer.ln1_weight, layer.ln1_bias);

    struct ggml_tensor * x_prev;

    if (sequence_length > 1) {
        x_prev = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embed, sequence_length);
        x_prev = ggml_set_1d_inplace(ctx, x_prev, state.att_xx, 0);
        x_prev = ggml_set_1d_inplace(
            ctx,
            x_prev,
            ggml_view_1d(ctx, x, n_embed * (sequence_length - 1), 0), n_embed * sizeof(float)
        );
    } else {
        x_prev = state.att_xx;
    }

    // sx = x - state.att_xx
    // xxx = x + sx * x_maa
    x_prev = ggml_sub_inplace(ctx, x_prev, x);
    struct ggml_tensor * xxx = ggml_add_inplace(
        ctx,
        ggml_mul(ctx, x_prev, layer.att_time_maa_x),
        x
    );

    // xxx = tanh(xxx @ tm_w1).view(5, 1, -1)
    xxx = ggml_reshape_4d(
        ctx,
        ggml_tanh_inplace(
            ctx,
            ggml_mul_mat(ctx, layer.att_time_maa_w1, xxx)
        ),
        layer.att_time_maa_w1->ne[1] / 5, 1, 5, sequence_length
    );

    xxx = ggml_cont(
        ctx,
        ggml_permute(ctx, xxx, 0, 1, 3, 2)
    );

    // xxx = torch.bmm(xxx, tm_w2)
    xxx = ggml_mul_mat(
        ctx,
        ggml_reshape_4d(
            ctx,
            layer.att_time_maa_w2,
            layer.att_time_maa_w2->ne[0], layer.att_time_maa_w2->ne[1], 1, 5
        ),
        xxx
    );

    xxx = ggml_reshape_2d(ctx, xxx, n_embed * sequence_length, 5);

    struct ggml_tensor * mw = ggml_reshape_2d(
        ctx,
        ggml_get_rows(ctx, xxx, ggml_new_i32(ctx, 0)),
        n_embed, sequence_length
    );
    struct ggml_tensor * mk = ggml_reshape_2d(
        ctx,
        ggml_get_rows(ctx, xxx, ggml_new_i32(ctx, 1)),
        n_embed, sequence_length
    );
    struct ggml_tensor * mv = ggml_reshape_2d(
        ctx,
        ggml_get_rows(ctx, xxx, ggml_new_i32(ctx, 2)),
        n_embed, sequence_length
    );
    struct ggml_tensor * mr = ggml_reshape_2d(
        ctx,
        ggml_get_rows(ctx, xxx, ggml_new_i32(ctx, 3)),
        n_embed, sequence_length
    );
    struct ggml_tensor * mg = ggml_reshape_2d(
        ctx,
        ggml_get_rows(ctx, xxx, ggml_new_i32(ctx, 4)),
        n_embed, sequence_length
    );

    struct ggml_tensor * xw = ggml_add_inplace(
        ctx,
        ggml_mul_inplace(
            ctx,
            ggml_add_inplace(ctx, mw, layer.att_time_maa_w),
            x_prev
        ),
        x
    );

    struct ggml_tensor * xk = ggml_add_inplace(
        ctx,
        ggml_mul_inplace(
            ctx,
            ggml_add_inplace(ctx, mk, layer.att_time_maa_k),
            x_prev
        ),
        x
    );

    struct ggml_tensor * xv = ggml_add_inplace(
        ctx,
        ggml_mul_inplace(
            ctx,
            ggml_add_inplace(ctx, mv, layer.att_time_maa_v),
            x_prev
        ),
        x
    );

    struct ggml_tensor * xr = ggml_add_inplace(
        ctx,
        ggml_mul_inplace(
            ctx,
            ggml_add_inplace(ctx, mr, layer.att_time_maa_r),
            x_prev
        ),
        x
    );

    struct ggml_tensor * xg = ggml_add_inplace(
        ctx,
        ggml_mul_inplace(
            ctx,
            ggml_add_inplace(ctx, mg, layer.att_time_maa_g),
            x_prev
        ),
        x
    );

    state.att_xx = ggml_view_1d(ctx, x, n_embed, n_embed * (sequence_length - 1) * sizeof(float));
    struct ggml_tensor * r = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, layer.att_receptance, xr), head_size, 1,         head_count, sequence_length);
    struct ggml_tensor * k = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, layer.att_key,        xk), 1,         head_size, head_count, sequence_length);
    struct ggml_tensor * v = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, layer.att_value,      xv), head_size, 1,         head_count, sequence_length);
    struct ggml_tensor * g = ggml_silu_inplace(
        ctx,
        ggml_mul_mat(ctx, layer.att_gate, xg)
    );

    struct ggml_tensor * w = ggml_mul_mat(
        ctx,
        layer.att_time_decay_w2,
        ggml_tanh_inplace(
            ctx,
            ggml_mul_mat(ctx, layer.att_time_decay_w1, xw)
        )
    );
    w = ggml_add_inplace(
        ctx,
        w,
        ggml_reshape_1d(ctx, layer.att_time_decay, n_embed)
    );

    w = rwkv_exp(ctx, ggml_neg_inplace(ctx, rwkv_exp(ctx, w)));
    w = ggml_reshape_4d(ctx, w, 1, head_size, head_count, sequence_length);

    // dup is not strictly required; doing it just in case.
    struct ggml_tensor * state_out = ggml_dup(ctx, state.att_heads);

    x = rwkv_wkv_v6(
        ctx,
        sequence_length,
        n_embed,
        head_count,
        head_size,
        x,
        k,
        v,
        r,
        layer.att_time_faaaa,
        w,
        state_out
    );

    state.att_heads = state_out;

    // rwkv/ggml ggml_group_norm uses eps=1e-5, while rwkv v6 uses eps=64e-5
    // Do 1/8 scale to x before group_norm for now.
    x = ggml_scale_inplace(ctx, x, 0.125);
    // ggml_group_norm considers groups in the third dimension.
    x = ggml_reshape_4d(ctx, x, 1, 1, n_embed, sequence_length);
    x = ggml_group_norm_inplace(ctx, x, head_count);
    // Convert back to a regular vector.
    x = ggml_reshape_2d(ctx, x, n_embed, sequence_length);
    x = ggml_add_inplace(
        ctx,
        ggml_mul_inplace(
            ctx,
            x,
            layer.att_ln_x_weight
        ),
        layer.att_ln_x_bias
    );

    x = ggml_mul_inplace(ctx, x, g);

    return ggml_mul_mat(ctx, layer.att_output, x);
}

static struct ggml_tensor * rwkv_ffn(struct ggml_context * ctx, struct ggml_tensor * x, struct rwkv_layer layer, struct rwkv_layer_state & state) {
    struct ggml_tensor * x_prev;
    rwkv_carry_x(ctx, layer.ln2_weight, layer.ln2_bias, x, x_prev, state.ffn_xx);

    // xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
    // xk = x * time_mix_k + state[5 * i + 0] * (1 - time_mix_k)
    struct ggml_tensor * xk = ggml_add_inplace(
        ctx,
        ggml_mul(ctx, x, layer.ffn_time_mix_k),
        ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.ffn_time_mix_k))
    );

    // xr = x * time_mix_r + state[5 * i + 0] * (1 - time_mix_r)
    struct ggml_tensor * xr = ggml_add_inplace(
        ctx,
        ggml_mul(ctx, x, layer.ffn_time_mix_r),
        ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.ffn_time_mix_r))
    );

    // r = torch.sigmoid(rw @ xr)
    struct ggml_tensor * r = rwkv_sigmoid_inplace(ctx, ggml_mul_mat(ctx, layer.ffn_receptance, xr));

    // k = torch.square(torch.relu(kw @ xk))
    struct ggml_tensor * k = ggml_sqr_inplace(ctx, ggml_relu_inplace(ctx, ggml_mul_mat(ctx, layer.ffn_key, xk)));

    // r * (vw @ k)
    return ggml_mul_inplace(ctx, r, ggml_mul_mat(ctx, layer.ffn_value, k));
}

static struct ggml_tensor * rwkv_ffn_v6(struct ggml_context * ctx, struct ggml_tensor * x, struct rwkv_layer layer, struct rwkv_layer_state & state) {
    struct ggml_tensor * x_prev;
    rwkv_carry_x(ctx, layer.ln2_weight, layer.ln2_bias, x, x_prev, state.ffn_xx);
    x_prev = ggml_sub_inplace(ctx, x_prev, x);

    // xk = x + sx * time_maa_k
    // xr = x + sx * time_maa_r
    struct ggml_tensor * xk = ggml_add_inplace(
        ctx,
        ggml_mul(ctx, x_prev, layer.ffn_time_maa_k),
        x
    );

    struct ggml_tensor * xr = ggml_add_inplace(
        ctx,
        ggml_mul(ctx, x_prev, layer.ffn_time_maa_r),
        x
    );

    // r = torch.sigmoid(rw @ xr)
    struct ggml_tensor * r = rwkv_sigmoid_inplace(ctx, ggml_mul_mat(ctx, layer.ffn_receptance, xr));

    // k = torch.square(torch.relu(kw @ xk))
    struct ggml_tensor * k = ggml_sqr_inplace(ctx, ggml_relu_inplace(ctx, ggml_mul_mat(ctx, layer.ffn_key, xk)));

    // r * (vw @ k)
    return ggml_mul_inplace(ctx, r, ggml_mul_mat(ctx, layer.ffn_value, k));
}

static void rwkv_create_input_and_output_views(
    struct ggml_context * ctx,
    struct rwkv_layer_state * inputs,
    struct rwkv_layer_state * outputs,
    struct ggml_tensor * input,
    struct ggml_tensor * output,
    const size_t n_layer,
    const size_t n_embed,
    const uint32_t arch_version_major,
    const int64_t head_count,
    const int64_t head_size
) {
    size_t sz_float = sizeof(float);

    for (size_t i = 0; i < n_layer; i++) {
        struct rwkv_layer_state & input_state = inputs[i];
        struct rwkv_layer_state & output_state = outputs[i];

        if (arch_version_major >= 5) {
            size_t vectors_per_layer = 2 + head_size;

            size_t att_heads_size = head_size * head_size * head_count;

            input_state.ffn_xx    = ggml_view_1d(ctx, input, n_embed,          n_embed * (i * vectors_per_layer + 0) * sz_float);
            input_state.att_xx    = ggml_view_1d(ctx, input, n_embed,          n_embed * (i * vectors_per_layer + 1) * sz_float);
            input_state.att_heads = ggml_view_1d(ctx, input, att_heads_size,   n_embed * (i * vectors_per_layer + 2) * sz_float);

            output_state.ffn_xx    = ggml_view_1d(ctx, output, n_embed,        n_embed * (i * vectors_per_layer + 0) * sz_float);
            output_state.att_xx    = ggml_view_1d(ctx, output, n_embed,        n_embed * (i * vectors_per_layer + 1) * sz_float);
            output_state.att_heads = ggml_view_1d(ctx, output, att_heads_size, n_embed * (i * vectors_per_layer + 2) * sz_float);
        } else {
            input_state.ffn_xx = ggml_view_1d(ctx, input, n_embed, n_embed * (i * 5 + 0) * sz_float);
            input_state.att_xx = ggml_view_1d(ctx, input, n_embed, n_embed * (i * 5 + 1) * sz_float);
            input_state.att_aa = ggml_view_1d(ctx, input, n_embed, n_embed * (i * 5 + 2) * sz_float);
            input_state.att_bb = ggml_view_1d(ctx, input, n_embed, n_embed * (i * 5 + 3) * sz_float);
            input_state.att_pp = ggml_view_1d(ctx, input, n_embed, n_embed * (i * 5 + 4) * sz_float);

            output_state.ffn_xx = ggml_view_1d(ctx, output, n_embed, n_embed * (i * 5 + 0) * sz_float);
            output_state.att_xx = ggml_view_1d(ctx, output, n_embed, n_embed * (i * 5 + 1) * sz_float);
            output_state.att_aa = ggml_view_1d(ctx, output, n_embed, n_embed * (i * 5 + 2) * sz_float);
            output_state.att_bb = ggml_view_1d(ctx, output, n_embed, n_embed * (i * 5 + 3) * sz_float);
            output_state.att_pp = ggml_view_1d(ctx, output, n_embed, n_embed * (i * 5 + 4) * sz_float);
        }

    }
}

// Serial graph (token-by-token eval)

// Creates and sets the input and output ggml tensors, builds the computation graph.
static bool rwkv_build_serial_graph(struct rwkv_model & model, struct rwkv_computation_graph & graph) {
    graph.cgraph.reset(rwkv_ggml_cgraph_create(RWKV_MAX_NODES, false));

    struct rwkv_file_header & header = model.header;
    const size_t n_vocab = header.n_vocab;
    const size_t n_embed = header.n_embed;
    const size_t n_layer = header.n_layer;

    struct ggml_context * ctx = graph.ggml_ctx;

    // Creates a 1-element tensor.
    graph.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    size_t vectors_per_layer = model.arch_version_major >= 5 ?
        2 + model.head_size :
        5;

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embed * vectors_per_layer * n_layer);
    struct ggml_tensor * output = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embed * vectors_per_layer * n_layer);

    // We collect parts of input state here. Each part is (n_embed) vector.
    std::unique_ptr<struct rwkv_layer_state[]> inputs(new(std::nothrow) struct rwkv_layer_state[n_layer]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, inputs.get(), "Failed to allocate input state parts");

    // We collect parts of output state here. Each part is (n_embed) vector.
    std::unique_ptr<struct rwkv_layer_state[]> outputs(new(std::nothrow) struct rwkv_layer_state[n_layer]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, outputs.get(), "Failed to allocate output state parts");

    rwkv_create_input_and_output_views(ctx, inputs.get(), outputs.get(), input, output, n_layer, n_embed, model.arch_version_major, model.head_count, model.head_size);

    graph.logits = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab);

    if (!graph.buffer_inputs) {
        size_t buffer_size = 1024; // overhead
        buffer_size += rwkv_tensor_nbytes(input);
        buffer_size += rwkv_tensor_nbytes(output);
        buffer_size += rwkv_tensor_nbytes(graph.logits);
        buffer_size += rwkv_tensor_nbytes(graph.tokens);
        graph.buffer_inputs = ggml_backend_alloc_buffer(model.backends.back(), buffer_size);
    }
    struct ggml_tallocr talloc = ggml_tallocr_new(graph.buffer_inputs);
    ggml_tallocr_alloc(&talloc, input);
    ggml_tallocr_alloc(&talloc, output);
    ggml_tallocr_alloc(&talloc, graph.logits);
    ggml_tallocr_alloc(&talloc, graph.tokens);

    // x = self.w.emb.weight[token]
    struct ggml_tensor * x = ggml_get_rows(ctx, model.emb, graph.tokens);

    // x = self.layer_norm(x, self.w.blocks[0].ln0)
    x = rwkv_layer_norm(ctx, x, model.ln0_weight, model.ln0_bias);

    for (size_t i = 0; i < n_layer; i++) {
        struct rwkv_layer & layer = model.layers[i];

        struct rwkv_layer_state state = inputs[i];

        if (model.arch_version_major == 6) {
            x = ggml_add_inplace(ctx, x, rwkv_att_v6(
                ctx,
                x,
                layer,
                state,
                model.head_count,
                model.head_size,
                model.arch_version_minor
            ));

            x = ggml_add_inplace(ctx, x, rwkv_ffn_v6(ctx, x, layer, state));
        } else {
            x = model.arch_version_major >= 5 ?
                ggml_add_inplace(ctx, x, rwkv_att_v5(
                    ctx,
                    x,
                    layer,
                    state,
                    model.head_count,
                    model.head_size,
                    model.arch_version_minor
                )) :
                ggml_add_inplace(ctx, x, rwkv_att(ctx, x, layer, state));

            x = ggml_add_inplace(ctx, x, rwkv_ffn(ctx, x, layer, state));
        }

        struct rwkv_layer_state & output_state = outputs[i];

        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.ffn_xx, output_state.ffn_xx));
        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_xx, output_state.att_xx));

        if (model.arch_version_major >= 5) {
            ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_heads, output_state.att_heads));
        } else {
            ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_aa, output_state.att_aa));
            ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_bb, output_state.att_bb));
            ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_pp, output_state.att_pp));
        }
    }

    graph.pre_logits_nodes = graph.cgraph->n_nodes;
    graph.pre_logits_leafs = graph.cgraph->n_leafs;

    // x = self.layer_norm(x[-1,:], self.w.ln_out)
    x = rwkv_layer_norm(ctx, x, model.ln_out_weight, model.ln_out_bias);

    // x = (self.w.head.weight @ x).float()
    ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, ggml_mul_mat(ctx, model.head, x), graph.logits));

    graph.post_logits_nodes = graph.cgraph->n_nodes;
    graph.post_logits_leafs = graph.cgraph->n_leafs;

    graph.input_state = input;
    graph.input_layers = std::move(inputs);

    graph.output_state = output;
    graph.output_layers = std::move(outputs);

    return true;
}

// Copy-pasted from llama.cpp.
static const size_t tensor_alignment = 32;

// Prepares the computation graph for inference, measuring and allocating all input and output tensors.
static bool rwkv_measure_and_build_serial_context(struct rwkv_model & model, struct rwkv_computation_graph & graph) {
    if (graph.ggml_ctx) {
        ggml_free(graph.ggml_ctx);

        graph.ggml_ctx = NULL;
    }

    graph.ggml_ctx = rwkv_init_ggml_context(rwkv_ggml_overhead(), true);

    RWKV_ENSURE_OR_FALSE(rwkv_build_serial_graph(model, graph));

    return true;
}

// Sequential graph

// Creates and sets the input and output ggml tensors, builds the computation graph.
static bool rwkv_build_sequential_graph(struct rwkv_model & model, struct rwkv_computation_graph & graph, const size_t sequence_length) {
    graph.cgraph.reset(rwkv_ggml_cgraph_create(RWKV_MAX_NODES, false));

    struct rwkv_file_header & header = model.header;
    const size_t n_vocab = header.n_vocab;
    const size_t n_embed = header.n_embed;
    const size_t n_layer = header.n_layer;

    struct ggml_context * ctx = graph.ggml_ctx;

    graph.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sequence_length);

    size_t vectors_per_layer = model.arch_version_major >= 5 ?
        2 + model.head_size :
        5;

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embed * vectors_per_layer * n_layer);
    struct ggml_tensor * output = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embed * vectors_per_layer * n_layer);

    // We collect parts of input state here. Each part is (n_embed) vector.
    std::unique_ptr<struct rwkv_layer_state[]> inputs(new(std::nothrow) struct rwkv_layer_state[n_layer]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, inputs.get(), "Failed to allocate input state parts");

    // We collect parts of output state here. Each part is (n_embed) vector.
    std::unique_ptr<struct rwkv_layer_state[]> outputs(new(std::nothrow) struct rwkv_layer_state[n_layer]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, outputs.get(), "Failed to allocate output state parts");

    rwkv_create_input_and_output_views(ctx, inputs.get(), outputs.get(), input, output, n_layer, n_embed, model.arch_version_major, model.head_count, model.head_size);

    graph.logits = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab);

    if (!graph.buffer_inputs) {
        size_t buffer_size = 1024; // overhead
        buffer_size += rwkv_tensor_nbytes(input);
        buffer_size += rwkv_tensor_nbytes(output);
        buffer_size += rwkv_tensor_nbytes(graph.logits);
        buffer_size += ggml_type_size(GGML_TYPE_I32) * 1024;
        graph.buffer_inputs = ggml_backend_alloc_buffer(model.backends.back(), buffer_size);
    }
    struct ggml_tallocr talloc = ggml_tallocr_new(graph.buffer_inputs);
    ggml_tallocr_alloc(&talloc, input);
    ggml_tallocr_alloc(&talloc, output);
    ggml_tallocr_alloc(&talloc, graph.logits);
    ggml_tallocr_alloc(&talloc, graph.tokens);

    // x = self.w.emb.weight[token]
    struct ggml_tensor * x = ggml_get_rows(ctx, model.emb, graph.tokens);

    // x = self.layer_norm(x, self.w.blocks[0].ln0)
    x = rwkv_layer_norm(ctx, x, ggml_repeat(ctx, model.ln0_weight, x), ggml_repeat(ctx, model.ln0_bias, x));

    for (size_t i = 0; i < model.header.n_layer; i++) {
        struct rwkv_layer & layer = model.layers[i];

        struct rwkv_layer_state state = inputs[i];

        if (model.arch_version_major == 6) {
            x = ggml_add_inplace(ctx, x, rwkv_att_v6(
                ctx,
                x,
                layer,
                state,
                model.head_count,
                model.head_size,
                model.arch_version_minor
            ));
        } else if (model.arch_version_major >= 5) {
            x = ggml_add_inplace(ctx, x, rwkv_att_v5(
                ctx,
                x,
                layer,
                state,
                model.head_count,
                model.head_size,
                model.arch_version_minor
            ));
        } else {
            struct ggml_tensor * x0 = x, * x_prev;
            rwkv_carry_x(ctx, layer.ln1_weight, layer.ln1_bias, x0, x_prev, state.att_xx);

            struct ggml_tensor * r, * k, * v;
            rwkv_att_rkv(ctx, layer, x0, x_prev, r, k, v);

            ggml_build_forward_expand(graph.cgraph.get(), r);

            for (size_t t = 0; t < sequence_length; t++) {
                struct ggml_tensor * kt = ggml_view_1d(ctx, k, n_embed, n_embed * sizeof(float) * t);
                struct ggml_tensor * vt = ggml_view_1d(ctx, v, n_embed, n_embed * sizeof(float) * t);
                struct ggml_tensor * xt = ggml_view_1d(ctx, x_prev, n_embed, n_embed * sizeof(float) * t);
                struct ggml_tensor * wkv = rwkv_att_wkv(ctx, layer.att_time_first, layer.att_time_decay, kt, vt, state.att_aa, state.att_bb, state.att_pp);
                ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, wkv, xt));
            }

            x = ggml_add_inplace(ctx, x, ggml_mul_mat(ctx, layer.att_output, ggml_mul(ctx, r, x_prev)));
        }

        // TODO Can we skip ffn for all but the last token, the same way we skip unembedding?
        if (model.arch_version_major == 6) {
            x = ggml_add_inplace(ctx, x, rwkv_ffn_v6(ctx, x, layer, state));
        } else {
            x = ggml_add_inplace(ctx, x, rwkv_ffn(ctx, x, layer, state));
        }

        struct rwkv_layer_state & output_state = outputs[i];

        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_xx, output_state.att_xx));
        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.ffn_xx, output_state.ffn_xx));

        if (model.arch_version_major >= 5) {
            ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_heads, output_state.att_heads));
        } else {
            ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_aa, output_state.att_aa));
            ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_bb, output_state.att_bb));
            ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_pp, output_state.att_pp));
        }
    }

    graph.pre_logits_nodes = graph.cgraph->n_nodes;
    graph.pre_logits_leafs = graph.cgraph->n_leafs;

    // x = self.layer_norm(x[-1,:], self.w.ln_out)
    x = rwkv_layer_norm(ctx, ggml_view_1d(ctx, x, n_embed, n_embed * sizeof(float) * (sequence_length - 1)), model.ln_out_weight, model.ln_out_bias);

    // x = (self.w.head.weight @ x).float()
    ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, ggml_mul_mat(ctx, model.head, x), graph.logits));

    graph.post_logits_nodes = graph.cgraph->n_nodes;
    graph.post_logits_leafs = graph.cgraph->n_leafs;

    graph.input_state = input;
    graph.input_layers = std::move(inputs);

    graph.output_state = output;
    graph.output_layers = std::move(outputs);

    return true;
}

// Prepares the computation graph for inference, measuring and allocating all input and output tensors.
static bool rwkv_measure_and_build_sequential_context(struct rwkv_model & model, struct rwkv_computation_graph & graph, const size_t sequence_length) {
    if (graph.ggml_ctx) {
        ggml_free(graph.ggml_ctx);

        graph.ggml_ctx = NULL;
    }

    graph.ggml_ctx = rwkv_init_ggml_context(rwkv_ggml_overhead(), true);

    RWKV_ENSURE_OR_FALSE(rwkv_build_sequential_graph(model, graph, sequence_length));

    return true;
}
