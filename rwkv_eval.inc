void rwkv_set_inputs(const struct rwkv_context * ctx, const struct rwkv_computation_graph & graph, const float * state_in) {
    if (state_in) {
        memcpy(graph.input_state->data, state_in, rwkv_tensor_nbytes(graph.input_state));
    } else {
        rwkv_init_state(ctx, (float *) graph.input_state->data);
    }
}

void rwkv_get_outputs(const struct rwkv_computation_graph & graph, float * state_out, float * logits_out) {
    if (state_out) {
        memcpy(state_out, graph.output_state->data, rwkv_tensor_nbytes(graph.output_state));
    }

    if (logits_out) {
        memcpy(logits_out, graph.logits->data, rwkv_tensor_nbytes(graph.logits));
    }
}

void rwkv_eval_graph(struct rwkv_computation_graph & graph, const uint32_t n_threads, const bool compute_logits) {
    // Short circuit computation of logits if they are not needed.
    if (!compute_logits) {
        graph.cgraph->n_nodes = graph.pre_logits_nodes;
        graph.cgraph->n_leafs = graph.pre_logits_leafs;
    } else {
        graph.cgraph->n_nodes = graph.post_logits_nodes;
        graph.cgraph->n_leafs = graph.post_logits_leafs;
    }

    struct ggml_cplan * plan = ggml_graph_plan(graph.cgraph.get(), n_threads);

    std::unique_ptr<uint8_t[]> work_data{ new(std::nothrow) uint8_t[plan->work_size] };
    plan->work_data = work_data.get();

    ggml_graph_compute(graph.cgraph.get(), plan);

    free(plan);
}

bool rwkv_eval(struct rwkv_context * ctx, const uint32_t token, const float * state_in, float * state_out, float * logits_out) {
    ctx->last_error = RWKV_ERROR_NONE;

    const struct rwkv_file_header & header = ctx->model->header;
    const size_t n_vocab = header.n_vocab;
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, token < n_vocab, "Token (%" PRId32 ") is out of range (0 .. %zu)", token, n_vocab - 1);

    rwkv_set_inputs(ctx, ctx->serial_graph, state_in);
    ggml_set_i32(ctx->serial_graph.tokens, token);

    rwkv_eval_graph(ctx->serial_graph, ctx->n_threads, logits_out != NULL);

    rwkv_get_outputs(ctx->serial_graph, state_out, logits_out);

    return true;
}

bool rwkv_eval_sequence(
    struct rwkv_context * ctx,
    const uint32_t * sequence,
    const size_t sequence_len,
    const float * state_in,
    float * state_out,
    float * logits_out
) {
    ctx->last_error = RWKV_ERROR_NONE;

    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, sequence_len > 0, "Sequence length is 0");

    const size_t n_vocab = ctx->model->header.n_vocab;

    if (sequence) {
        for (size_t i = 0; i < sequence_len; i++) {
            const uint32_t token = sequence[i];

            RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, token < n_vocab, "Token at index %zu (%" PRId32 ") is out of range (0 .. %zu)", i, token, n_vocab - 1);
        }
    }

    if (ctx->last_used_sequence_length != sequence_len) {
        RWKV_ENSURE_OR_FALSE(rwkv_measure_and_build_sequential_context(*ctx->model, ctx->sequential_graph, sequence_len));

        ctx->last_used_sequence_length = sequence_len;
    }

    // Allow building the sequence graph without actually evaluating, by specifying sequence = NULL.
    if (sequence) {
        rwkv_set_inputs(ctx, ctx->sequential_graph, state_in);
        memcpy(ctx->sequential_graph.tokens->data, sequence, sequence_len * sizeof(uint32_t));

        rwkv_eval_graph(ctx->sequential_graph, ctx->n_threads, logits_out != NULL);

        rwkv_get_outputs(ctx->sequential_graph, state_out, logits_out);
    }

    return true;
}

void rwkv_init_state(const struct rwkv_context * ctx, float * state) {
    const struct rwkv_file_header & header = ctx->model->header;
    const size_t layer_size = (size_t) header.n_embed * 5;
    const size_t layer_zero = (size_t) header.n_embed * 4;
    const size_t layers_size = (size_t) header.n_layer * layer_size;

    for (size_t start = 0; start < layers_size; start += layer_size) {
        for (size_t i = 0; i < layer_zero; i++) {
            state[start + i] = 0.0F;
        }

        for (size_t i = layer_zero; i < layer_size; i++) {
            state[start + i] = -1e30F;
        }
    }
}
