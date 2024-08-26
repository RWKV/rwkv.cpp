struct rwkv_layer {
    struct ggml_tensor * ln1_weight;
    struct ggml_tensor * ln1_bias;

    // RWKV, also called "attention" by the author.
    struct ggml_tensor * att_time_mix_k;
    struct ggml_tensor * att_time_mix_v;
    struct ggml_tensor * att_time_mix_r;
    // Removed in RWKV v5.2; set to NULL for this and newer models.
    struct ggml_tensor * att_time_first;
    struct ggml_tensor * att_time_decay;
    struct ggml_tensor * att_key;
    struct ggml_tensor * att_value;
    struct ggml_tensor * att_receptance;
    struct ggml_tensor * att_output;

    // Added in RWKV v5.1; set to NULL for earlier models (v4).
    struct ggml_tensor * att_ln_x_weight;
    struct ggml_tensor * att_ln_x_bias;

    // Added in RWKV v5.2; set to NULL for earlier models (v4, v5.1).
    struct ggml_tensor * att_time_faaaa;
    struct ggml_tensor * att_time_mix_g;
    struct ggml_tensor * att_gate;

    // Added in RWKV v6.
    struct ggml_tensor * att_time_maa_x;
    struct ggml_tensor * att_time_maa_w;
    struct ggml_tensor * att_time_maa_k;
    struct ggml_tensor * att_time_maa_v;
    struct ggml_tensor * att_time_maa_r;
    struct ggml_tensor * att_time_maa_g;
    struct ggml_tensor * att_time_maa_w1;
    struct ggml_tensor * att_time_maa_w2;
    struct ggml_tensor * att_time_decay_w1;
    struct ggml_tensor * att_time_decay_w2;

    struct ggml_tensor * ln2_weight;
    struct ggml_tensor * ln2_bias;

    // FFN.
    struct ggml_tensor * ffn_time_mix_k;
    struct ggml_tensor * ffn_time_mix_r;

    // Added in RWKV v6.
    struct ggml_tensor * ffn_time_maa_k;
    struct ggml_tensor * ffn_time_maa_r;

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

    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_t> buffers_w;
    std::vector<ggml_tallocr> tallocrs;

    struct rwkv_file_header header;
    uint32_t arch_version_major;
    uint32_t arch_version_minor;
    // Added in RWKV v5.1; set to 0 for earlier models (v4).
    int64_t head_count;
    int64_t head_size;

    uint32_t rescale_every_n_layers = 999;

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
static bool rwkv_set_params(struct rwkv_model & model, F callback, const uint32_t n_gpu_layers) {
    const size_t n_gpu = std::min(n_gpu_layers, model.header.n_layer + 1);
    bool offload_head = n_gpu == (model.header.n_layer + 1);
    bool offload_default = false;

    RWKV_ENSURE_OR_FALSE(callback("emb.weight", model.emb, offload_default));
    RWKV_ENSURE_OR_FALSE(callback("blocks.0.ln0.weight", model.ln0_weight, (n_gpu_layers > 0)));
    RWKV_ENSURE_OR_FALSE(callback("blocks.0.ln0.bias", model.ln0_bias, (n_gpu_layers > 0)));

    uint32_t n_layer = model.header.n_layer;
    std::unique_ptr<struct rwkv_layer[]> layers(new(std::nothrow) struct rwkv_layer[n_layer]());
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, layers.get(), "Failed to allocate model layers");
    model.layers = std::move(layers);

    for (uint32_t i = 0; i < n_layer; i++) {
        bool offload_layer = (i < n_gpu);
        char buffer[128];
        size_t offset = sprintf(buffer, "blocks.%" PRId32 ".", i);

        rwkv_layer & layer = model.layers[i];
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln1.weight"), buffer), layer.ln1_weight, offload_layer));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln1.bias"), buffer), layer.ln1_bias, offload_layer));

        if (model.arch_version_major == 6) {
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_maa_x"), buffer), layer.att_time_maa_x, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_maa_w"), buffer), layer.att_time_maa_w, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_maa_k"), buffer), layer.att_time_maa_k, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_maa_v"), buffer), layer.att_time_maa_v, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_maa_r"), buffer), layer.att_time_maa_r, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_maa_g"), buffer), layer.att_time_maa_g, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_maa_w1"), buffer), layer.att_time_maa_w1, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_maa_w2"), buffer), layer.att_time_maa_w2, offload_layer));

            // No gpu offloading for wkv yet
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_faaaa"), buffer), layer.att_time_faaaa, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_decay"), buffer), layer.att_time_decay, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_decay_w1"), buffer), layer.att_time_decay_w1, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_decay_w2"), buffer), layer.att_time_decay_w2, offload_default));

            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.key.weight"), buffer), layer.att_key, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.value.weight"), buffer), layer.att_value, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.receptance.weight"), buffer), layer.att_receptance, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.gate.weight"), buffer), layer.att_gate, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.output.weight"), buffer), layer.att_output, offload_layer));

            // GroupNorm uses a custom epsilon value, which only has CPU implementation for now.
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.ln_x.weight"), buffer), layer.att_ln_x_weight, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.ln_x.bias"), buffer), layer.att_ln_x_bias, offload_default));
        } else {
            // Custom rwkv_1_minus_x: cpu only
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_mix_k"), buffer), layer.att_time_mix_k, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_mix_v"), buffer), layer.att_time_mix_v, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_mix_r"), buffer), layer.att_time_mix_r, offload_default));

            if (model.arch_version_major >= 5 && model.arch_version_minor >= 2) {
                RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_faaaa"), buffer), layer.att_time_faaaa, offload_default));
            } else {
                RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_first"), buffer), layer.att_time_first, offload_default));
            }

            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_decay"), buffer), layer.att_time_decay, offload_default));

            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.key.weight"), buffer), layer.att_key, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.value.weight"), buffer), layer.att_value, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.receptance.weight"), buffer), layer.att_receptance, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.output.weight"), buffer), layer.att_output, offload_layer));

            if (model.arch_version_major >= 5) {
                RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.ln_x.weight"), buffer), layer.att_ln_x_weight, offload_default));
                RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.ln_x.bias"), buffer), layer.att_ln_x_bias, offload_default));

                if (model.arch_version_minor >= 2) {
                    RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.time_mix_g"), buffer), layer.att_time_mix_g, offload_default));
                    RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "att.gate.weight"), buffer), layer.att_gate, offload_layer));
                }
            }
        }


        if (model.arch_version_major == 6) {
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln2.weight"), buffer), layer.ln2_weight, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln2.bias"), buffer), layer.ln2_bias, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.time_maa_k"), buffer), layer.ffn_time_maa_k, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.time_maa_r"), buffer), layer.ffn_time_maa_r, offload_layer));
        } else {
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln2.weight"), buffer), layer.ln2_weight, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ln2.bias"), buffer), layer.ln2_bias, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.time_mix_k"), buffer), layer.ffn_time_mix_k, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.time_mix_r"), buffer), layer.ffn_time_mix_r, offload_default));
        }

        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.key.weight"), buffer), layer.ffn_key, offload_layer));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.value.weight"), buffer), layer.ffn_value, offload_layer));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "ffn.receptance.weight"), buffer), layer.ffn_receptance, offload_layer));
    }

    RWKV_ENSURE_OR_FALSE(callback("ln_out.weight", model.ln_out_weight, offload_head));
    RWKV_ENSURE_OR_FALSE(callback("ln_out.bias", model.ln_out_bias, offload_head));
    RWKV_ENSURE_OR_FALSE(callback("head.weight", model.head, offload_head));

    return true;
}

template<typename F>
static bool rwkv_set_params_gguf(struct rwkv_model & model, F callback, const uint32_t n_gpu_layers) {
    const size_t n_gpu = std::min(n_gpu_layers, model.header.n_layer + 1);
    bool offload_head = n_gpu == (model.header.n_layer + 1);
    bool offload_default = false;

    RWKV_ENSURE_OR_FALSE(callback("token_embd.weight", model.emb, offload_default));
    RWKV_ENSURE_OR_FALSE(callback("token_embd_norm.weight", model.ln0_weight, (n_gpu_layers > 0)));
    RWKV_ENSURE_OR_FALSE(callback("token_embd_norm.bias", model.ln0_bias, (n_gpu_layers > 0)));

    uint32_t n_layer = model.header.n_layer;
    std::unique_ptr<struct rwkv_layer[]> layers(new(std::nothrow) struct rwkv_layer[n_layer]());
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, layers.get(), "Failed to allocate model layers");
    model.layers = std::move(layers);

    for (uint32_t i = 0; i < n_layer; i++) {
        bool offload_layer = (i < n_gpu);
        char buffer[128];
        size_t offset = sprintf(buffer, "blk.%" PRId32 ".", i);

        rwkv_layer & layer = model.layers[i];
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "attn_norm.weight"), buffer), layer.ln1_weight, offload_layer));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "attn_norm.bias"), buffer), layer.ln1_bias, offload_layer));

        if (model.arch_version_major == 6) {
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_lerp_x.weight"), buffer), layer.att_time_maa_x, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_lerp_w.weight"), buffer), layer.att_time_maa_w, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_lerp_k.weight"), buffer), layer.att_time_maa_k, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_lerp_v.weight"), buffer), layer.att_time_maa_v, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_lerp_r.weight"), buffer), layer.att_time_maa_r, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_lerp_g.weight"), buffer), layer.att_time_maa_g, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_w1.weight"), buffer), layer.att_time_maa_w1, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_w2.weight"), buffer), layer.att_time_maa_w2, offload_layer));

            // No gpu offloading for wkv yet
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_first.weight"), buffer), layer.att_time_faaaa, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_decay.weight"), buffer), layer.att_time_decay, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_decay_w1.weight"), buffer), layer.att_time_decay_w1, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_decay_w2.weight"), buffer), layer.att_time_decay_w2, offload_default));

            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_key.weight"), buffer), layer.att_key, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_value.weight"), buffer), layer.att_value, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_receptance.weight"), buffer), layer.att_receptance, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_gate.weight"), buffer), layer.att_gate, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_output.weight"), buffer), layer.att_output, offload_layer));

            // GroupNorm uses a custom epsilon value, which only has CPU implementation for now.
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_ln.weight"), buffer), layer.att_ln_x_weight, offload_default));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "time_mix_ln.bias"), buffer), layer.att_ln_x_bias, offload_default));

            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "attn_norm_2.weight"), buffer), layer.ln2_weight, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "attn_norm_2.bias"), buffer), layer.ln2_bias, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "channel_mix_lerp_k.weight"), buffer), layer.ffn_time_maa_k, offload_layer));
            RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "channel_mix_lerp_r.weight"), buffer), layer.ffn_time_maa_r, offload_layer));
        } else {
            return false;
        }

        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "channel_mix_key.weight"), buffer), layer.ffn_key, offload_layer));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "channel_mix_value.weight"), buffer), layer.ffn_value, offload_layer));
        RWKV_ENSURE_OR_FALSE(callback((strcpy(&buffer[offset], "channel_mix_receptance.weight"), buffer), layer.ffn_receptance, offload_layer));
    }

    RWKV_ENSURE_OR_FALSE(callback("output_norm.weight", model.ln_out_weight, offload_head));
    RWKV_ENSURE_OR_FALSE(callback("output_norm.bias", model.ln_out_bias, offload_head));
    RWKV_ENSURE_OR_FALSE(callback("output.weight", model.head, offload_head));

    return true;
}

#include "ggml.h"
#include <iostream>

// Creates a ggml context and loads all parameter tensors from a model file.
static bool rwkv_load_model_from_bin_file(const char * file_path, struct rwkv_model & model, const uint32_t n_gpu_layers) {
    struct stat file_stat;

    std::unordered_map<std::string, struct ggml_tensor *> parameters;

    rwkv_file file(fopen(file_path, "rb"));

    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_OPEN, file.file, "Failed to open file %s", file_path);
    // Be very careful when changing this code. It must support files larger than 2 GB by using 64-bit functions to get the file length.
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_STAT, fstat(fileno(file.file), &file_stat) == 0, "Failed to stat file %s", file_path);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE, rwkv_fread_file_header(file.file, model.header), "Invalid file header");

    model.ggml_ctx = rwkv_init_ggml_context(
        rwkv_ggml_overhead(),
        true // no-alloc; allocate tensors in different backend buffers later
    );

    std::string name;

    struct ggml_tensor * tensor;

    // Read all tensor information from the file first.
    auto tensors_file_start = ftell(file.file);
    while ((size_t) ftell(file.file) < (size_t) file_stat.st_size) {
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS, 
            rwkv_fread_ggml_tensor_info(file.file, model.ggml_ctx, name, tensor), // dry_run = true
            "Failed to read a model parameter");

        parameters[std::move(name)] = tensor;
    }

    model.arch_version_major = 4;
    model.arch_version_minor = 0;

    if (parameters.find("blocks.0.att.ln_x.weight") != parameters.end()) {
        model.arch_version_major = 5;

        if (parameters.find("blocks.0.att.gate.weight") != parameters.end()) {
            model.arch_version_minor = 2;
        } else {
            model.arch_version_minor = 1;
        }
    }

    if (parameters.find("blocks.0.att.time_maa_x") != parameters.end()) {
        model.arch_version_major = 6;
        model.arch_version_minor = 0;
    }

    size_t cpu_buffer_size = 0;
    size_t gpu_buffer_size = 0;
    std::unordered_map<std::string, struct ggml_tensor *> & parameters_ref = parameters;
    // Calculate buffer sizes for each backend.
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_PARAM_MISSING, rwkv_set_params(
        model,
        [&](const char * key, struct ggml_tensor *& dest, bool offload_gpu) {
            struct ggml_tensor * tensor = parameters_ref[key];
            RWKV_ENSURE_OR_FALSE_MSG(tensor, "Model parameter %s not found", key);
            if (offload_gpu && n_gpu_layers)
                gpu_buffer_size += ggml_nbytes(tensor);
            else
                cpu_buffer_size += ggml_nbytes(tensor);
            dest = tensor;
            return true;
        },
        n_gpu_layers
    ));

    cpu_buffer_size += ggml_tensor_overhead() * RWKV_MAX_NODES;
    if (n_gpu_layers) {
        gpu_buffer_size += ggml_tensor_overhead() * RWKV_MAX_NODES;
    }

    // Allocate buffers for each backend.
    if (n_gpu_layers) {
        ggml_backend_t backend_gpu = model.backends.front();
        ggml_backend_buffer_t gpu_buffer = ggml_backend_alloc_buffer(backend_gpu, gpu_buffer_size);
        ggml_backend_buffer_set_usage(gpu_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        model.buffers_w.push_back(gpu_buffer);
        model.tallocrs.push_back(ggml_tallocr_new(gpu_buffer));
    }

    ggml_backend_t backend_cpu = model.backends.back();
    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_buffer(backend_cpu, cpu_buffer_size);
    ggml_backend_buffer_set_usage(cpu_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    model.buffers_w.push_back(cpu_buffer);
    model.tallocrs.push_back(ggml_tallocr_new(cpu_buffer));

    // Allocate tensors in backend buffers.
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_PARAM_MISSING, rwkv_set_params(
        model,
        [&](const char * key, struct ggml_tensor *& dest, bool offload_gpu) {
            struct ggml_tensor * tensor = parameters_ref[key];
            RWKV_ENSURE_OR_FALSE_MSG(tensor, "Model parameter %s not found", key);
            ggml_tallocr * alloc = offload_gpu ? &model.tallocrs.front() : &model.tallocrs.back();
            ggml_tallocr_alloc(alloc, tensor);
            dest = tensor;
            return true;
        },
        n_gpu_layers
    ));

    // Read tensor data.
    fseek(file.file, tensors_file_start, SEEK_SET);
    while ((size_t) ftell(file.file) < (size_t) file_stat.st_size) {
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS, 
            rwkv_fread_ggml_tensor_data(file.file, model.ggml_ctx, parameters_ref),
            "Failed to read a model parameter");
    }

    if (model.arch_version_major >= 5) {
        model.head_count = model.layers[0].att_time_decay->ne[2];
        model.head_size = model.layers[0].ln1_weight->ne[0] / model.head_count;
    }

    // Verify order of dimensions.
    struct ggml_tensor * emb = model.emb;
    int n_dims = ggml_n_dims(emb);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_SHAPE, n_dims == 2, "Unexpected dimension count of embedding matrix %d", n_dims);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, emb->ne[0] == model.header.n_embed, "Unexpected dimension of embedding matrix %" PRId64, emb->ne[0]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, emb->ne[1] == model.header.n_vocab, "Unexpected dimension of embedding matrix %" PRId64, emb->ne[1]);

    return true;
}

static bool rwkv_gguf_get_key_u32(gguf_context * gguf_ctx, const char * key, uint32_t & value) {
    int key_id = gguf_find_key(gguf_ctx, key);
    if (key_id < 0) {
        return false;
    }
    value = gguf_get_val_u32(gguf_ctx, key_id);
    return true;
}

static bool rwkv_load_model_from_gguf_file(const char * file_path, struct rwkv_model & model, const uint32_t n_gpu_layers) {
    gguf_context * gguf_ctx = gguf_init_from_file(file_path, {true, &model.ggml_ctx});

    // int n_kv = gguf_get_n_kv(gguf_ctx);
    int n_tensors = gguf_get_n_tensors(gguf_ctx);
    std::string arch = std::string(gguf_get_val_str(gguf_ctx, gguf_find_key(gguf_ctx, "general.architecture")));
    if (arch == "rwkv6") {
        model.arch_version_major = 6;
        model.arch_version_minor = 0;
    } else {
        // gguf only supports RWKV v6 for now
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL, false, "Unsupported architecture %s", arch.c_str());
    }

    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL, rwkv_gguf_get_key_u32(gguf_ctx, (arch + ".embedding_length").c_str(), model.header.n_embed), "Failed to get n_embed");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL, rwkv_gguf_get_key_u32(gguf_ctx, (arch + ".block_count").c_str(), model.header.n_layer), "Failed to get n_layer");
    if (!rwkv_gguf_get_key_u32(gguf_ctx, (arch + ".rescale_every_n_layers").c_str(), model.rescale_every_n_layers)) {
        model.rescale_every_n_layers = 999;
    }

    if (!rwkv_gguf_get_key_u32(gguf_ctx, (arch + ".vocab_size").c_str(), model.header.n_vocab)) {
        int key_id = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
        if (key_id >= 0) {
            model.header.n_vocab = gguf_get_arr_n(gguf_ctx, key_id);
        } else {
            RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL, false, "Failed to get n_vocab");
        }
    }

    size_t cpu_buffer_size = 0;
    size_t gpu_buffer_size = 0;
    // Calculate buffer sizes for each backend.
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_PARAM_MISSING, rwkv_set_params_gguf(
        model,
        [&](const char * key, struct ggml_tensor *& dest, bool offload_gpu) {
            struct ggml_tensor * tensor = nullptr, *cur;
            for (cur = ggml_get_first_tensor(model.ggml_ctx); cur; cur = ggml_get_next_tensor(model.ggml_ctx, cur)) {
                if (strcmp(ggml_get_name(cur), key) == 0) {
                    tensor = cur;
                    break;
                }
            }
            RWKV_ENSURE_OR_FALSE_MSG(tensor, "Model parameter %s not found", key);
            if (offload_gpu && n_gpu_layers)
                gpu_buffer_size += ggml_nbytes(tensor);
            else
                cpu_buffer_size += ggml_nbytes(tensor);
            dest = tensor;
            return true;
        },
        n_gpu_layers
    ));

    cpu_buffer_size += ggml_tensor_overhead() * RWKV_MAX_NODES;
    if (n_gpu_layers) {
        gpu_buffer_size += ggml_tensor_overhead() * RWKV_MAX_NODES;
    }

    // Allocate buffers for each backend.
    if (n_gpu_layers) {
        ggml_backend_t backend_gpu = model.backends.front();
        ggml_backend_buffer_t gpu_buffer = ggml_backend_alloc_buffer(backend_gpu, gpu_buffer_size);
        ggml_backend_buffer_set_usage(gpu_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        model.buffers_w.push_back(gpu_buffer);
        model.tallocrs.push_back(ggml_tallocr_new(gpu_buffer));
    }

    ggml_backend_t backend_cpu = model.backends.back();
    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_buffer(backend_cpu, cpu_buffer_size);
    ggml_backend_buffer_set_usage(cpu_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    model.buffers_w.push_back(cpu_buffer);
    model.tallocrs.push_back(ggml_tallocr_new(cpu_buffer));

    gguf_free(gguf_ctx);

    ggml_context * ggml_ctx;
    gguf_ctx = gguf_init_from_file(file_path, {false, &ggml_ctx});

    // Allocate tensors in backend buffers.
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_PARAM_MISSING, rwkv_set_params_gguf(
        model,
        [&](const char * key, struct ggml_tensor *& dest, bool offload_gpu) {
            struct ggml_tensor * tensor = nullptr;
            struct ggml_tensor * tensor_gguf = nullptr;
            struct ggml_tensor * cur;
            for (cur = ggml_get_first_tensor(ggml_ctx); cur; cur = ggml_get_next_tensor(ggml_ctx, cur)) {
                if (strcmp(ggml_get_name(cur), key) == 0) {
                    tensor_gguf = cur;
                    break;
                }
            }
            for (cur = ggml_get_first_tensor(model.ggml_ctx); cur; cur = ggml_get_next_tensor(model.ggml_ctx, cur)) {
                if (strcmp(ggml_get_name(cur), key) == 0) {
                    tensor = cur;
                    break;
                }
            }

            RWKV_ENSURE_OR_FALSE_MSG(tensor && tensor_gguf, "Model parameter %s not found", key);
            ggml_tallocr * alloc = offload_gpu ? &model.tallocrs.front() : &model.tallocrs.back();
            ggml_tallocr_alloc(alloc, tensor);
            dest = tensor;
            ggml_backend_tensor_set(tensor, tensor_gguf->data, 0, ggml_nbytes(tensor));
            return true;
        },
        n_gpu_layers
    ));

    model.head_count = model.layers[0].att_time_faaaa->ne[1];
    model.head_size = model.layers[0].ln1_weight->ne[0] / model.head_count;

    gguf_free(gguf_ctx);
    ggml_free(ggml_ctx);

    return true;
}

static bool rwkv_load_model_from_file(const char * file_path, struct rwkv_model & model, const uint32_t n_gpu_layers) {
    std::string file_path_str(file_path);
    size_t dot_pos = file_path_str.find_last_of('.');
    if (dot_pos != std::string::npos) {
        std::string extension = file_path_str.substr(dot_pos + 1);
        if (extension == "bin") {
            return rwkv_load_model_from_bin_file(file_path, model, n_gpu_layers);
        } else if (extension == "gguf") {
            return rwkv_load_model_from_gguf_file(file_path, model, n_gpu_layers);
        }
    } else {
        // try the legacy format anyway
        return rwkv_load_model_from_bin_file(file_path, model, n_gpu_layers);
    }
}