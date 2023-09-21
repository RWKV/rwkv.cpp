struct rwkv_layer {
    struct ggml_tensor * ln1_weight;
    struct ggml_tensor * ln1_bias;

    // RWKV, also called "attention" by the author.
    struct ggml_tensor * att_time_mix_k;
    struct ggml_tensor * att_time_mix_v;
    struct ggml_tensor * att_time_mix_r;
    struct ggml_tensor * att_time_first;
    struct ggml_tensor * att_time_decay;
    struct ggml_tensor * att_key;
    struct ggml_tensor * att_value;
    struct ggml_tensor * att_receptance;
    struct ggml_tensor * att_output;

    struct ggml_tensor * ln2_weight;
    struct ggml_tensor * ln2_bias;

    // FFN.
    struct ggml_tensor * ffn_time_mix_k;
    struct ggml_tensor * ffn_time_mix_r;
    struct ggml_tensor * ffn_key;
    struct ggml_tensor * ffn_value;
    struct ggml_tensor * ffn_receptance;
};

// The model holds all parameter tensors and the ggml context containing them.
// Each tensor has data and can be used in computations happening in other contexts.
struct rwkv_model {
    // This context holds all parameter tensors.
    // It must not be used for computations.
    struct ggml_context * ggml_ctx;

    struct rwkv_file_header header;

    struct ggml_tensor * emb;

    struct ggml_tensor * ln0_weight;
    struct ggml_tensor * ln0_bias;

    std::unique_ptr<struct rwkv_layer[]> layers;

    struct ggml_tensor * ln_out_weight;
    struct ggml_tensor * ln_out_bias;

    struct ggml_tensor * head;

    // How many layers were offloaded to the GPU.
    // Model head is counted as an additional layer,
    // so the max value for this field is n_layers + 1.
    size_t offloaded_layer_count;

    // How many RWKV contexts reference this model.
    int reference_count;
};

struct rwkv_file {
    FILE * file;

    rwkv_file(FILE * file): file(file) {}

    ~rwkv_file() {
        if (file) {
            fclose(file);
        }
    }
};

// https://stackoverflow.com/a/6458689
template<typename F>
static bool rwkv_set_params(struct rwkv_model & model, F callback) {
    RWKV_ENSURE_OR_FALSE(callback("emb.weight", model.emb));
    RWKV_ENSURE_OR_FALSE(callback("blocks.0.ln0.weight", model.ln0_weight));
    RWKV_ENSURE_OR_FALSE(callback("blocks.0.ln0.bias", model.ln0_bias));

    uint32_t n_layer = model.header.n_layer;
    std::unique_ptr<struct rwkv_layer[]> layers(new(std::nothrow) struct rwkv_layer[n_layer]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, layers.get(), "Failed to allocate model layers");
    model.layers = std::move(layers);

    for (uint32_t i = 0; i < n_layer; i++) {
        char buffer[128];
        size_t offset = sprintf(buffer, "blocks.%" PRId32 ".", i);

        rwkv_layer & layer = model.layers[i];
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln1.weight"), buffer), layer.ln1_weight));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln1.bias"), buffer), layer.ln1_bias));

        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_mix_k"), buffer), layer.att_time_mix_k));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_mix_v"), buffer), layer.att_time_mix_v));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_mix_r"), buffer), layer.att_time_mix_r));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_first"), buffer), layer.att_time_first));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_decay"), buffer), layer.att_time_decay));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.key.weight"), buffer), layer.att_key));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.value.weight"), buffer), layer.att_value));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.receptance.weight"), buffer), layer.att_receptance));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.output.weight"), buffer), layer.att_output));

        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln2.weight"), buffer), layer.ln2_weight));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln2.bias"), buffer), layer.ln2_bias));

        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.time_mix_k"), buffer), layer.ffn_time_mix_k));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.time_mix_r"), buffer), layer.ffn_time_mix_r));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.key.weight"), buffer), layer.ffn_key));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.value.weight"), buffer), layer.ffn_value));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.receptance.weight"), buffer), layer.ffn_receptance));
    }

    RWKV_ENSURE_OR_FALSE(callback("ln_out.weight", model.ln_out_weight));
    RWKV_ENSURE_OR_FALSE(callback("ln_out.bias", model.ln_out_bias));
    RWKV_ENSURE_OR_FALSE(callback("head.weight", model.head));

    return true;
}

// Creates a ggml context and loads all parameter tensors from a model file.
static bool rwkv_load_model_from_file(const char * file_path, struct rwkv_model & model) {
    struct stat file_stat;

    std::unordered_map<std::string, struct ggml_tensor *> parameters;

    rwkv_file file(fopen(file_path, "rb"));

    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_OPEN, file.file, "Failed to open file %s", file_path);
    // Be very careful when changing this code. It must support files larger than 2 GB by using 64-bit functions to get the file length.
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_STAT, fstat(fileno(file.file), &file_stat) == 0, "Failed to stat file %s", file_path);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE, rwkv_fread_file_header(file.file, model.header), "Invalid file header");

    model.ggml_ctx = rwkv_init_ggml_context(
        // ggml tensors must be aligned; assuming here that overhead of parameter headers, included in the file size, will account for that.
        file_stat.st_size + rwkv_ggml_overhead(),
        false
    );

    std::string name;

    struct ggml_tensor * tensor;

    while ((size_t) ftell(file.file) < (size_t) file_stat.st_size) {
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS, rwkv_fread_ggml_tensor(file.file, model.ggml_ctx, name, tensor), "Failed to read a model parameter");

        parameters[std::move(name)] = tensor;
    }

    std::unordered_map<std::string, struct ggml_tensor *> & parameters_ref = parameters;
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_PARAM_MISSING, rwkv_set_params(model, [&](const char * key, struct ggml_tensor *& dest) {
        struct ggml_tensor * tensor = parameters_ref[key];
        RWKV_ENSURE_OR_FALSE_MSG(tensor, "Model parameter %s not found", key);
        dest = tensor;
        return true;
    }));

    // Verify order of dimensions.
    struct ggml_tensor * emb = model.emb;
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_SHAPE, emb->n_dims == 2, "Unexpected dimension count of embedding matrix %d", emb->n_dims);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, emb->ne[0] == model.header.n_embed, "Unexpected dimension of embedding matrix %" PRId64, emb->ne[0]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, emb->ne[1] == model.header.n_vocab, "Unexpected dimension of embedding matrix %" PRId64, emb->ne[1]);

    return true;
}
