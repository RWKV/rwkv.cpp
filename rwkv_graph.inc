// View tensors of a state of a single layer.
struct rwkv_layer_state {
    struct ggml_tensor * ffn_xx;
    struct ggml_tensor * att_xx;
    struct ggml_tensor * att_aa;
    struct ggml_tensor * att_bb;
    struct ggml_tensor * att_pp;
};

// The computation graph holds ggml context and the ggml cgraph.
// It can be either a serial or a sequential graph.
struct rwkv_computation_graph {
    struct ggml_context * ggml_ctx;
    // ggml_cgraph is so large that it can cause stack overflows if not stored on the heap.
    std::unique_ptr<struct ggml_cgraph> cgraph;

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
        x = rwkv_layer_norm(ctx, x, ggml_repeat(ctx, weight, x), ggml_repeat(ctx, bias, x));

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
    r = rwkv_sigmoid(ctx, ggml_mul_mat(ctx, layer.att_receptance, xr));
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
    struct ggml_tensor * r = rwkv_sigmoid(ctx, ggml_mul_mat(ctx, layer.ffn_receptance, xr));

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
    const size_t n_embed
) {
    for (size_t i = 0; i < n_layer; i++) {
        struct rwkv_layer_state & input_state = inputs[i];
        input_state.ffn_xx = ggml_view_1d(ctx, input, n_embed, n_embed * (i * 5 + 0) * sizeof(float));
        input_state.att_xx = ggml_view_1d(ctx, input, n_embed, n_embed * (i * 5 + 1) * sizeof(float));
        input_state.att_aa = ggml_view_1d(ctx, input, n_embed, n_embed * (i * 5 + 2) * sizeof(float));
        input_state.att_bb = ggml_view_1d(ctx, input, n_embed, n_embed * (i * 5 + 3) * sizeof(float));
        input_state.att_pp = ggml_view_1d(ctx, input, n_embed, n_embed * (i * 5 + 4) * sizeof(float));

        struct rwkv_layer_state & output_state = outputs[i];
        output_state.ffn_xx = ggml_view_1d(ctx, output, n_embed, n_embed * (i * 5 + 0) * sizeof(float));
        output_state.att_xx = ggml_view_1d(ctx, output, n_embed, n_embed * (i * 5 + 1) * sizeof(float));
        output_state.att_aa = ggml_view_1d(ctx, output, n_embed, n_embed * (i * 5 + 2) * sizeof(float));
        output_state.att_bb = ggml_view_1d(ctx, output, n_embed, n_embed * (i * 5 + 3) * sizeof(float));
        output_state.att_pp = ggml_view_1d(ctx, output, n_embed, n_embed * (i * 5 + 4) * sizeof(float));
    }
}

// Serial graph (token-by-token eval)

// Creates and sets the input and output ggml tensors, builds the computation graph.
static bool rwkv_build_serial_graph(struct rwkv_model & model, struct rwkv_computation_graph & graph) {
    graph.cgraph.reset(new(std::nothrow) struct ggml_cgraph());

    struct rwkv_file_header & header = model.header;
    const size_t n_vocab = header.n_vocab;
    const size_t n_embed = header.n_embed;
    const size_t n_layer = header.n_layer;

    struct ggml_context * ctx = graph.ggml_ctx;

    // Creates a 1-element tensor.
    graph.tokens = ggml_new_i32(ctx, 0);

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embed * 5 * n_layer);
    struct ggml_tensor * output = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embed * 5 * n_layer);

    // We collect parts of input state here. Each part is (n_embed) vector.
    std::unique_ptr<struct rwkv_layer_state[]> inputs(new(std::nothrow) struct rwkv_layer_state[n_layer]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, inputs.get(), "Failed to allocate input state parts");

    // We collect parts of output state here. Each part is (n_embed) vector.
    std::unique_ptr<struct rwkv_layer_state[]> outputs(new(std::nothrow) struct rwkv_layer_state[n_layer]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, outputs.get(), "Failed to allocate output state parts");

    rwkv_create_input_and_output_views(ctx, inputs.get(), outputs.get(), input, output, n_layer, n_embed);

    graph.logits = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab);

    // x = self.w.emb.weight[token]
    struct ggml_tensor * x = ggml_get_rows(ctx, model.emb, graph.tokens);

    // x = self.layer_norm(x, self.w.blocks[0].ln0)
    x = rwkv_layer_norm(ctx, x, model.ln0_weight, model.ln0_bias);

    for (size_t i = 0; i < model.header.n_layer; i++) {
        struct rwkv_layer & layer = model.layers[i];

        struct rwkv_layer_state state = inputs[i];
        x = ggml_add_inplace(ctx, x, rwkv_att(ctx, x, layer, state));
        x = ggml_add_inplace(ctx, x, rwkv_ffn(ctx, x, layer, state));

        struct rwkv_layer_state & output_state = outputs[i];
        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.ffn_xx, output_state.ffn_xx));
        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_xx, output_state.att_xx));
        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_aa, output_state.att_aa));
        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_bb, output_state.att_bb));
        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_pp, output_state.att_pp));
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

    // 1. Measure the space required for the ggml context.
    graph.ggml_ctx = rwkv_init_ggml_context(rwkv_ggml_overhead(), true);

    RWKV_ENSURE_OR_FALSE(rwkv_build_serial_graph(model, graph));

    struct ggml_allocr * allocator = ggml_allocr_new_measure(tensor_alignment);

    size_t required_context_size = ggml_allocr_alloc_graph(allocator, graph.cgraph.get()) +
            // With the node limit set 80K, this overhead would be 28 MB.
            + rwkv_ggml_overhead()
            + tensor_alignment
            // For some reason, `ggml_allocr_alloc_graph` underestimates required memory amount.
            // Instead of diving deep into ggml internals to debug this issue, I will just add some padding.
            // 40 MB seems to be enough for Raven 14B model when GGML_MAX_NODES is set to default value of 4096.
            + size_t(40) * 1024 * 1024;

    ggml_allocr_free(allocator);
    ggml_free(graph.ggml_ctx);

    // 2. Create the real ggml context.
    graph.ggml_ctx = rwkv_init_ggml_context(required_context_size, false);

    RWKV_ENSURE_OR_FALSE(rwkv_build_serial_graph(model, graph));

    return true;
}

// Sequential graph

// Creates and sets the input and output ggml tensors, builds the computation graph.
static bool rwkv_build_sequential_graph(struct rwkv_model & model, struct rwkv_computation_graph & graph, const size_t sequence_length) {
    graph.cgraph.reset(new(std::nothrow) struct ggml_cgraph());

    struct rwkv_file_header & header = model.header;
    const size_t n_vocab = header.n_vocab;
    const size_t n_embed = header.n_embed;
    const size_t n_layer = header.n_layer;

    struct ggml_context * ctx = graph.ggml_ctx;

    graph.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sequence_length);

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embed * 5 * n_layer);
    struct ggml_tensor * output = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embed * 5 * n_layer);

    // We collect parts of input state here. Each part is (n_embed) vector.
    std::unique_ptr<struct rwkv_layer_state[]> inputs(new(std::nothrow) struct rwkv_layer_state[n_layer]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, inputs.get(), "Failed to allocate input state parts");

    // We collect parts of output state here. Each part is (n_embed) vector.
    std::unique_ptr<struct rwkv_layer_state[]> outputs(new(std::nothrow) struct rwkv_layer_state[n_layer]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, outputs.get(), "Failed to allocate output state parts");

    rwkv_create_input_and_output_views(ctx, inputs.get(), outputs.get(), input, output, n_layer, n_embed);

    graph.logits = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab);

    // x = self.w.emb.weight[token]
    struct ggml_tensor * x = ggml_get_rows(ctx, model.emb, graph.tokens);

    // x = self.layer_norm(x, self.w.blocks[0].ln0)
    x = rwkv_layer_norm(ctx, x, ggml_repeat(ctx, model.ln0_weight, x), ggml_repeat(ctx, model.ln0_bias, x));

    for (size_t i = 0; i < model.header.n_layer; i++) {
        struct rwkv_layer & layer = model.layers[i];
        struct rwkv_layer_state state = inputs[i];

        struct ggml_tensor * x0 = x, * x_prev;
        rwkv_carry_x(ctx, layer.ln1_weight, layer.ln1_bias, x0, x_prev, state.att_xx);

        struct ggml_tensor * r, * k, * v;
        rwkv_att_rkv(ctx, layer, x0, x_prev, r, k, v);

        ggml_build_forward_expand(graph.cgraph.get(), r);

        for (uint32_t t = 0; t < sequence_length; t++) {
            struct ggml_tensor * kt = ggml_view_1d(ctx, k, n_embed, n_embed * sizeof(float) * t);
            struct ggml_tensor * vt = ggml_view_1d(ctx, v, n_embed, n_embed * sizeof(float) * t);
            struct ggml_tensor * xt = ggml_view_1d(ctx, x_prev, n_embed, n_embed * sizeof(float) * t);
            struct ggml_tensor * wkv = rwkv_att_wkv(ctx, layer.att_time_first, layer.att_time_decay, kt, vt, state.att_aa, state.att_bb, state.att_pp);
            ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, wkv, xt));
        }

        x = ggml_add_inplace(ctx, x, ggml_mul_mat(ctx, layer.att_output, ggml_mul(ctx, r, x_prev)));
        x = ggml_add_inplace(ctx, x, rwkv_ffn(ctx, x, layer, state));

        struct rwkv_layer_state & output_state = outputs[i];
        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.ffn_xx, output_state.ffn_xx));
        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_xx, output_state.att_xx));
        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_aa, output_state.att_aa));
        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_bb, output_state.att_bb));
        ggml_build_forward_expand(graph.cgraph.get(), ggml_cpy(ctx, state.att_pp, output_state.att_pp));
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

    // 1. Measure the space required for the ggml context.
    graph.ggml_ctx = rwkv_init_ggml_context(rwkv_ggml_overhead(), true);

    RWKV_ENSURE_OR_FALSE(rwkv_build_sequential_graph(model, graph, sequence_length));

    struct ggml_allocr * allocator = ggml_allocr_new_measure(tensor_alignment);

    size_t required_context_size = ggml_allocr_alloc_graph(allocator, graph.cgraph.get()) +
            // With the node limit set 80K, this overhead would be 28 MB.
            + rwkv_ggml_overhead()
            + tensor_alignment
            // For some reason, `ggml_allocr_alloc_graph` underestimates required memory amount.
            // Instead of diving deep into ggml internals to debug this issue, I will just add some padding.
            // 40 MB per token seems to be enough for Raven 14B model. It works for sequence_length at least up to 71.
            + sequence_length * 40 * 1024 * 1024;

    ggml_allocr_free(allocator);
    ggml_free(graph.ggml_ctx);

    // 2. Create the real ggml context.
    graph.ggml_ctx = rwkv_init_ggml_context(required_context_size, false);

    RWKV_ENSURE_OR_FALSE(rwkv_build_sequential_graph(model, graph, sequence_length));

    return true;
}
