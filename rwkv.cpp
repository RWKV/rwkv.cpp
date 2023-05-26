#include "rwkv.h"
#include "ggml.h"

#include <string>
#include <vector>
#include <cstring>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <memory>

#include <sys/stat.h> // fstat

#define RWKV_MAYBE_BREAK

#ifdef WIN32
#define stat64 _stat64
#define fstat64 _fstat64
#define ftell64 _ftelli64
#define fseek64 _fseeki64

#ifndef NDEBUG
#include <intrin.h>
#define RWKV_MAYBE_BREAK __debugbreak()
#endif
#else
#define ftell64 ftello64
#define fseek64 fseeko64
#endif

// --- Error handling ---

thread_local enum rwkv_error_flags global_last_error = RWKV_ERROR_NONE;
thread_local bool global_print_errors = true;

inline enum rwkv_error_flags operator|(enum rwkv_error_flags a, enum rwkv_error_flags b) {
    return static_cast<enum rwkv_error_flags>(static_cast<int>(a) | static_cast<int>(b));
}

inline enum rwkv_error_flags operator|=(enum rwkv_error_flags & a, enum rwkv_error_flags b) {
    return a = a | b;
}

#define RWKV_MSG(...) do { if (global_print_errors) fprintf(stderr, __VA_ARGS__); } while (0)
#define RWKV_CTX_MSG(ctx, ...) do { if ((ctx)->print_errors) fprintf(stderr, __VA_ARGS__); } while (0)

// If the condition x is false, adds ERR_VAL to the last error, and returns RET_VAL.
#define RWKV_ASSERT(ERR_VAL, RET_VAL, x) \
    do { if (!(x)) { global_last_error |= ERR_VAL; RWKV_MAYBE_BREAK; return RET_VAL; } } while (0)

// If the condition x is false, adds ERR_VAL to the last error, prints a message to stderr, and returns RET_VAL.
#define RWKV_ASSERT_MSG(ERR_VAL, RET_VAL, x, ...) do { \
    if (!(x)) { \
        global_last_error |= ERR_VAL; \
        RWKV_MSG(__VA_ARGS__); \
        RWKV_MSG("\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

// If the condition x is false, adds ERR_VAL to the ctx's last error, prints a message to stderr, and returns RET_VAL.
#define RWKV_CTX_ASSERT_MSG(ctx, ERR_VAL, RET_VAL, x, ...) do { \
    if (!(x)) { \
        ((struct rwkv_context *) ctx)->last_error |= ERR_VAL; \
        RWKV_CTX_MSG(ctx, __VA_ARGS__); \
        RWKV_CTX_MSG(ctx, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

// If the condition x is false, adds ERR_VAL to the ctx's last error, and returns RET_VAL.
#define RWKV_CTX_ASSERT(ctx, ERR_VAL, RET_VAL, x) \
    do { if (!(x)) { ((struct rwkv_context *) ctx)->last_error |= ERR_VAL; RWKV_MAYBE_BREAK; return RET_VAL; } } while (0)

// If the condition x is false, returns RET_VAL.
#define RWKV_ENSURE(RET_VAL, x) \
    do { if (!(x)) { RWKV_MAYBE_BREAK; return RET_VAL; } } while (0)

// If the condition x is false, prints a message to stderr, and returns RET_VAL.
#define RWKV_ENSURE_MSG(RET_VAL, x, ...) do { \
    if (!(x)) { \
        RWKV_MSG(__VA_ARGS__); \
        RWKV_MSG("\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

// If the condition x is false, prints a message to stderr, and returns RET_VAL.
#define RWKV_CTX_ENSURE_MSG(ctx, RET_VAL, x, ...) do { \
    if (!(x)) { \
        ((struct rwkv_context *) ctx)->last_error |= ERR_VAL; \
        RWKV_CTX_MSG(ctx, __VA_ARGS__); \
        RWKV_CTX_MSG(ctx, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

#define RWKV_ASSERT_FALSE_MSG(ERR_VAL, x, ...) RWKV_ASSERT_MSG(ERR_VAL, false, x, __VA_ARGS__)
#define RWKV_ASSERT_NULL_MSG(ERR_VAL, x, ...) RWKV_ASSERT_MSG(ERR_VAL, NULL, x, __VA_ARGS__)
#define RWKV_CTX_ASSERT_FALSE_MSG(ctx, ERR_VAL, x, ...) RWKV_CTX_ASSERT_MSG(ctx, ERR_VAL, false, x, __VA_ARGS__)
#define RWKV_CTX_ASSERT_NULL_MSG(ctx, ERR_VAL, x, ...) RWKV_CTX_ASSERT_MSG(ctx, ERR_VAL, NULL, x, __VA_ARGS__)

#define RWKV_ASSERT_FALSE(ERR_VAL, x) RWKV_ASSERT(ERR_VAL, false, x)
#define RWKV_ASSERT_NULL(ERR_VAL, x) RWKV_ASSERT(ERR_VAL, NULL, x)
#define RWKV_CTX_ASSERT_FALSE(ctx, ERR_VAL, x) RWKV_CTX_ASSERT(ctx, ERR_VAL, false, x)
#define RWKV_CTX_ASSERT_NULL(ctx, ERR_VAL, x) RWKV_CTX_ASSERT(ctx, ERR_VAL, NULL, x)

#define RWKV_ENSURE_OR_FALSE(x) RWKV_ENSURE(false, x)
#define RWKV_ENSURE_OR_NULL(x) RWKV_ENSURE(NULL, x)
#define RWKV_ENSURE_OR_FALSE_MSG(x, ...) RWKV_ENSURE_MSG(false, x, __VA_ARGS__)
#define RWKV_ENSURE_OR_NULL_MSG(x, ...) RWKV_ENSURE_MSG(NULL, x, __VA_ARGS__)
#define RWKV_CTX_ENSURE_OR_FALSE_MSG(ctx, x, ...) RWKV_CTX_ENSURE_MSG(ctx, false, x, __VA_ARGS__)
#define RWKV_CTX_ENSURE_OR_NULL_MSG(ctx, x, ...) RWKV_CTX_ENSURE_MSG(ctx, NULL, x, __VA_ARGS__)

// --- Utilities ---

// Reads a single int32 value from a file.
bool fread_int32(FILE * file, int32_t & dest) {
    return fread((void *) &dest, sizeof(int32_t), 1, file) == 1;
}

// Reads a single uint32 value from a file.
bool fread_uint32(FILE * file, uint32_t & dest) {
    return fread((void *) &dest, sizeof(uint32_t), 1, file) == 1;
}

// Reads a single string value from a file.
bool fread_string(FILE * file, size_t length, std::string & dest) {
    dest.resize(length);
    return fread((void *) dest.data(), length, 1, file) == 1;
}

// Reads a single data buffer from a file.
bool fread_data(FILE * file, size_t length, void * dest) {
    return fread(dest, length, 1, file) == 1;
}

// Writes a single int32 value to a file.
bool fwrite_int32(FILE * file, const int32_t value) {
    return fwrite((const void *) &value, sizeof(int32_t), 1, file);
}

// Writes a single uint32 value to a file.
bool fwrite_uint32(FILE * file, const uint32_t value) {
    return fwrite((const void *) &value, sizeof(uint32_t), 1, file);
}

// Writes a single string value to a file.
bool fwrite_string(FILE * file, const std::string & value) {
    return fwrite((const void *) value.data(), value.length(), 1, file) == 1;
}

// Writes a single data buffer to a file.
bool fwrite_data(FILE * file, const void * data, const size_t length) {
    return fwrite(data, length, 1, file) == 1;
}

// --- File data structures ---

#define TYPE_UNKNOWN TYPE_COUNT

enum rwkv_type {
    TYPE_F32,
    TYPE_F16,
    TYPE_Q4_0,
    TYPE_Q4_1,
    TYPE_Q4_1_O, // Unsupported
    TYPE_Q4_2,
    TYPE_Q4_3, // Unsupported
    TYPE_Q5_0,
    TYPE_Q5_1,
    TYPE_Q8_0,
    TYPE_COUNT
};

#define GGML_TYPE_UNKNOWN GGML_TYPE_COUNT

static const enum ggml_type type_to_ggml[TYPE_COUNT + 1] = {
    GGML_TYPE_F32,     /* F32    */
    GGML_TYPE_F16,     /* F16    */
    GGML_TYPE_Q4_0,    /* Q4_0   */
    GGML_TYPE_Q4_1,    /* Q4_1   */
    GGML_TYPE_UNKNOWN, /* Q4_1_O */
    GGML_TYPE_Q4_2,    /* Q4_2   */
    GGML_TYPE_UNKNOWN, /* Q4_3   */
    GGML_TYPE_Q5_0,    /* Q5_0   */
    GGML_TYPE_Q5_1,    /* Q5_1   */
    GGML_TYPE_Q8_0,    /* Q8_0   */
    GGML_TYPE_COUNT    /* COUNT  */
};

static const enum rwkv_type type_from_ggml[GGML_TYPE_COUNT + 1] = {
    TYPE_F32,    /* F32   */
    TYPE_F16,    /* F16   */
    TYPE_Q4_0,   /* Q4_0  */
    TYPE_Q4_1,   /* Q4_1  */
    TYPE_Q4_2,   /* Q4_2  */
    TYPE_Q4_3,   /* Q4_3  */
    TYPE_Q5_0,   /* Q5_0  */
    TYPE_Q5_1,   /* Q5_1  */
    TYPE_Q8_0,   /* Q8_0  */
    TYPE_COUNT,  /* Q8_1  */
    TYPE_COUNT,  /* I8    */
    TYPE_COUNT,  /* I16   */
    TYPE_COUNT,  /* I32   */
    TYPE_COUNT,  /* COUNT */
};

static const char * type_to_string[TYPE_COUNT] = {"F32", "F16", "Q4_0", "Q4_1", "Q4_1_O", "Q4_2", "Q4_3", "Q5_0", "Q5_1", "Q8_0"};

static enum rwkv_type type_from_string(const char * str) {
    for (int ord = 0; ord < TYPE_COUNT; ord++)
        if (strcmp(str, type_to_string[ord]) == 0)
            return (enum rwkv_type) ord;

    return TYPE_UNKNOWN;
}

struct file_header {
    uint32_t magic;
    uint32_t version;
    uint32_t n_vocab;
    uint32_t n_embed;
    uint32_t n_layer;
    uint32_t data_type;
};

static bool fread_file_header(FILE * file, struct file_header & header) {
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, fread_uint32(file, header.magic));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_MAGIC, header.magic == RWKV_FILE_MAGIC);
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, fread_uint32(file, header.version));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, fread_uint32(file, header.n_vocab));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, fread_uint32(file, header.n_embed));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, fread_uint32(file, header.n_layer));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, fread_uint32(file, header.data_type));
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_DATA_TYPE, header.data_type < TYPE_COUNT, "model data type out of range (%" PRId32 " > %" PRId32 ")", header.data_type, TYPE_COUNT - 1);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_DATA_TYPE, type_to_ggml[header.data_type] != GGML_TYPE_UNKNOWN, "model data type (%s) is no longer supported", type_to_string[header.data_type]);
    return true;
}

static bool fwrite_file_header(FILE * file, const struct file_header & header) {
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, fwrite_uint32(file, header.magic));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, fwrite_uint32(file, header.version));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, fwrite_uint32(file, header.n_vocab));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, fwrite_uint32(file, header.n_embed));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, fwrite_uint32(file, header.n_layer));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, fwrite_uint32(file, header.data_type));
    return true;
}

struct tensor_header {
    uint32_t dim_count;
    uint32_t key_length;
    uint32_t data_type;
    uint32_t width;
    uint32_t height;
};

static bool fread_tensor_header(FILE * file, struct tensor_header & header) {
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, fread_uint32(file, header.dim_count));
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_SHAPE, header.dim_count == 1 || header.dim_count == 2, "tensor has an invalid shape (%" PRId32 " dimensions)", header.dim_count);
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, fread_uint32(file, header.key_length));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, fread_uint32(file, header.data_type));
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_DATA_TYPE, header.data_type < TYPE_COUNT, "tensor data type out of range (%" PRId32 " > %" PRId32 ")", header.data_type, TYPE_COUNT - 1);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_DATA_TYPE, type_to_ggml[header.data_type] != GGML_TYPE_UNKNOWN, "tensor data type (%s) is no longer supported", type_to_string[header.data_type]);
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, fread_uint32(file, header.width));

    if (header.dim_count == 2)
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, fread_uint32(file, header.height));
    else
        header.height = 1;

    return true;
}

static bool fwrite_tensor_header(FILE * file, const struct tensor_header & header) {
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, fwrite_uint32(file, header.dim_count));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, fwrite_uint32(file, header.key_length));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, fwrite_uint32(file, header.data_type));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, fwrite_uint32(file, header.width));

    if (header.dim_count == 2)
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, fwrite_uint32(file, header.height));

    return true;
}

static size_t tensor_bytes(enum ggml_type type, const int64_t width, const int64_t height) {
    static struct ggml_tensor decoy {};
    decoy.type = type;
    decoy.ne[0] = width;
    decoy.ne[1] = height;
    decoy.ne[2] = 1;
    decoy.ne[3] = 1;
    return ggml_nbytes(&decoy);
}

static size_t tensor_bytes(const struct tensor_header & header) {
    return tensor_bytes(type_to_ggml[header.data_type], header.width, header.height);
}

static bool fskip_tensor_data(FILE * file, const struct tensor_header & header) {
    return fseek(file, header.key_length + tensor_bytes(header), SEEK_CUR) == 0;
}

// --- Model definition ---

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

struct rwkv_model {
    struct file_header header;

    struct ggml_tensor * emb;

    struct ggml_tensor * ln0_weight;
    struct ggml_tensor * ln0_bias;

    std::vector<rwkv_layer> layers;

    struct ggml_tensor * ln_out_weight;
    struct ggml_tensor * ln_out_bias;

    struct ggml_tensor * head;
};

// --- Operators ---

void rwkv_exp_impl(const int n_cols, float * dest, const float * src) {
    for (int i = 0; i < n_cols; i++) {
        dest[i] = expf(src[i]);
    }
}

void rwkv_1_minus_x_impl(const int n_cols, float * dest, const float * src) {
    for (int i = 0; i < n_cols; i++) {
        dest[i] = 1.0F - src[i];
    }
}

void rwkv_sigmoid_impl(const int n_cols, float * dest, const float * src) {
    for (int i = 0; i < n_cols; i++) {
        dest[i] = 1.0F / (1.0F + expf(-src[i]));
    }
}

void rwkv_max_impl(const int n_cols, float * dest, const float * src0, const float * src1) {
    for (int i = 0; i < n_cols; i++) {
        dest[i] = fmaxf(src0[i], src1[i]);
    }
}

struct ggml_tensor * rwkv_exp(ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_map_unary_f32(ctx, x, rwkv_exp_impl);
}

struct ggml_tensor * rwkv_1_minus_x(ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_map_unary_f32(ctx, x, rwkv_1_minus_x_impl);
}

struct ggml_tensor * rwkv_sigmoid(ggml_context * ctx, struct ggml_tensor * x) {
    return ggml_map_unary_f32(ctx, x, rwkv_sigmoid_impl);
}

struct ggml_tensor * rwkv_max(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y) {
    return ggml_map_binary_f32(ctx, x, y, rwkv_max_impl);
}

struct ggml_tensor * rwkv_layer_norm(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * weight, struct ggml_tensor * bias) {
    // LayerNorm in RWKV is `x = (x - mean(x)) / sqrt(variance(x) + 1e-5) * weight + bias`
    // Looks like ggml_norm does the first part, we only need to apply weight & bias.
    return ggml_add_inplace(ctx, ggml_mul(ctx, ggml_norm(ctx, x), weight), bias);
}

// --- Implementation ---

struct rwkv_graph {
    struct ggml_tensor * state;
    std::unique_ptr<struct ggml_tensor * []> state_parts;
    struct ggml_tensor * token_index;
    struct ggml_tensor * logits;
    std::unique_ptr<struct ggml_cgraph> cgraph;
};

struct rwkv_context {
    struct rwkv_model model;
    struct ggml_context * ctx;
    struct rwkv_graph graph;
    enum rwkv_error_flags last_error;
    bool print_errors;
};

bool fread_tensor_key(FILE * file, const struct tensor_header & header, const char * dest) {
    return fread_data(file, header.key_length, (void *) dest);
}

bool fread_tensor_data(FILE * file, const struct tensor_header & header, void * dest) {
    return fread_data(file, tensor_bytes(header), dest);
}

bool fread_ggml_tensor_data(FILE * file, const struct tensor_header & header, struct ggml_context * ctx, std::string & name, struct ggml_tensor *& tensor) {
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_READ, fread_string(file, header.key_length, name), "failed to read tensor name");

    enum ggml_type ggml_type = type_to_ggml[header.data_type];
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_UNSUPPORTED, ggml_type != GGML_TYPE_UNKNOWN, "unsupported tensor data type %s from %s", type_to_string[header.data_type], name.c_str());

    tensor = header.dim_count == 1
        ? ggml_new_tensor_1d(ctx, ggml_type, header.width)
        : ggml_new_tensor_2d(ctx, ggml_type, header.width, header.height);

    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, tensor, "failed to allocate tensor");
    ggml_set_name(tensor, name.c_str());

    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_READ, fread_data(file, ggml_nbytes(tensor), tensor->data), "failed to read tensor data from %s", name.c_str());
    return true;
}

bool fread_ggml_tensor(FILE * file, struct ggml_context * ctx, std::string & name, struct ggml_tensor *& tensor) {
    struct tensor_header header;
    RWKV_ENSURE_OR_FALSE_MSG(fread_tensor_header(file, header), "invalid tensor header");
    return fread_ggml_tensor_data(file, header, ctx, name, tensor);
}

template<typename F> // https://stackoverflow.com/a/6458689
bool rwkv_set_params(struct rwkv_model & model, F callback) {
    RWKV_ENSURE_OR_FALSE(callback("emb.weight", model.emb));
    RWKV_ENSURE_OR_FALSE(callback("blocks.0.ln0.weight", model.ln0_weight));
    RWKV_ENSURE_OR_FALSE(callback("blocks.0.ln0.bias", model.ln0_bias));

    uint32_t n_layer = model.header.n_layer;
    std::vector<struct rwkv_layer> & layers = model.layers;
    layers.resize(n_layer);

    for (uint32_t i = 0; i < n_layer; i++) {
        char buffer[128];
        size_t offset = sprintf(buffer, "blocks.%" PRId32 ".", i);

        rwkv_layer & layer = layers[i];
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

bool rwkv_build_graph(struct ggml_context * ctx, struct rwkv_model & model, const uint32_t n_threads, struct rwkv_graph & out) {
    std::unique_ptr<struct ggml_cgraph> cgraph(new(std::nothrow) struct ggml_cgraph());
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, cgraph.get(), "Failed to allocate graph");
    cgraph->n_threads = n_threads;

    size_t n_embed = model.header.n_embed, n_layer = model.header.n_layer;
    struct ggml_tensor * state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_layer * 5 * n_embed);

    // We collect parts of new state here. Each part is (n_embed) vector.
    std::unique_ptr<struct ggml_tensor * []> state_parts(new(std::nothrow) ggml_tensor * [n_layer * 5]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, state_parts.get(), "Failed to allocate state parts");

    // x = self.w.emb.weight[token]
    struct ggml_tensor * token_index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    struct ggml_tensor * x = ggml_get_rows(ctx, model.emb, token_index);

    // x = self.layer_norm(x, self.w.blocks[0].ln0)
    x = rwkv_layer_norm(ctx, x, model.ln0_weight, model.ln0_bias);

    for (size_t i = 0; i < n_layer; i++) {
        struct rwkv_layer layer = model.layers[i];
        size_t part_index = i * 5;
        size_t state_part_size = n_embed * sizeof(float);

        // RWKV/time mixing
        {
            // self.layer_norm(x, self.w.blocks[i].ln1)
            struct ggml_tensor * x0 = rwkv_layer_norm(ctx, x, layer.ln1_weight, layer.ln1_bias);

            // x0 = state[5 * i + 1]
            struct ggml_tensor * x_prev = ggml_view_1d(ctx, state, n_embed, (part_index + 1) * state_part_size);
            // aa = state[5 * i + 2]
            struct ggml_tensor * aa = ggml_view_1d(ctx, state, n_embed, (part_index + 2) * state_part_size);
            // bb = state[5 * i + 3]
            struct ggml_tensor * bb = ggml_view_1d(ctx, state, n_embed, (part_index + 3) * state_part_size);
            // pp = state[5 * i + 4]
            struct ggml_tensor * pp = ggml_view_1d(ctx, state, n_embed, (part_index + 4) * state_part_size);

            // xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
            struct ggml_tensor * xk = ggml_add_inplace(ctx,
                ggml_mul(ctx, x0, layer.att_time_mix_k),
                ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.att_time_mix_k))
            );

            // xv = x * time_mix_v + state[5 * i + 1] * (1 - time_mix_v)
            struct ggml_tensor * xv = ggml_add_inplace(ctx,
                ggml_mul(ctx, x0, layer.att_time_mix_v),
                ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.att_time_mix_v))
            );

            // xr = x * time_mix_r + state[5 * i + 1] * (1 - time_mix_r)
            struct ggml_tensor * xr = ggml_add_inplace(ctx,
                ggml_mul(ctx, x0, layer.att_time_mix_r),
                ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.att_time_mix_r))
            );

            // r = torch.sigmoid(rw @ xr)
            struct ggml_tensor * r = rwkv_sigmoid(ctx, ggml_mul_mat(ctx, layer.att_receptance, xr));
            // k = kw @ xk
            struct ggml_tensor * k = ggml_mul_mat(ctx, layer.att_key, xk);
            // v = vw @ xv
            struct ggml_tensor * v = ggml_mul_mat(ctx, layer.att_value, xv);

            // ww = time_first + k
            struct ggml_tensor * ww = ggml_add(ctx, layer.att_time_first, k);
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
            ww = ggml_add_inplace(ctx, pp, layer.att_time_decay);
            // qq = torch.maximum(ww, k)
            qq = rwkv_max(ctx, ww, k);
            // e1 = torch.exp(ww - qq)
            e1 = rwkv_exp(ctx, ggml_sub(ctx, ww, qq));
            // e2 = torch.exp(k - qq)
            e2 = rwkv_exp(ctx, ggml_sub(ctx, k, qq));

            // state[5 * i + 1] = x0
            // state[5 * i + 2] = e1 * aa + e2 * v
            // state[5 * i + 3] = e1 * bb + e2
            // state[5 * i + 4] = qq

            state_parts[part_index + 1] = x0;
            state_parts[part_index + 2] = ggml_add_inplace(ctx, ggml_mul(ctx, e1, aa), ggml_mul(ctx, e2, v));
            state_parts[part_index + 3] = ggml_add_inplace(ctx, ggml_mul(ctx, e1, bb), e2);
            state_parts[part_index + 4] = qq;

            // wkv = a / b
            struct ggml_tensor * wkv = ggml_div(ctx, a, b);

            // ow @ (r * wkv)
            x = ggml_add_inplace(ctx, x, ggml_mul_mat(ctx, layer.att_output, ggml_mul(ctx, r, wkv)));
        }

        // FFN/channel mixing
        {
            // self.layer_norm(x, self.w.blocks[i].ln2)
            struct ggml_tensor * x0 = rwkv_layer_norm(ctx, x, layer.ln2_weight, layer.ln2_bias);

            // x_prev = state[5 * i + 0]
            struct ggml_tensor * x_prev = ggml_view_1d(ctx, state, n_embed, part_index * state_part_size);

            // xk = x * time_mix_k + state[5 * i + 0] * (1 - time_mix_k)
            struct ggml_tensor * xk = ggml_add_inplace(
                ctx,
                ggml_mul(ctx, x0, layer.ffn_time_mix_k),
                ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.ffn_time_mix_k))
            );

            // xr = x * time_mix_r + state[5 * i + 0] * (1 - time_mix_r)
            struct ggml_tensor * xr = ggml_add_inplace(
                ctx,
                ggml_mul(ctx, x0, layer.ffn_time_mix_r),
                ggml_mul(ctx, x_prev, rwkv_1_minus_x(ctx, layer.ffn_time_mix_r))
            );

            // state[5 * i + 0] = x
            state_parts[part_index] = x0;

            // r = torch.sigmoid(rw @ xr)
            struct ggml_tensor * r = rwkv_sigmoid(ctx, ggml_mul_mat(ctx, layer.ffn_receptance, xr));

            // k = torch.square(torch.relu(kw @ xk))
            struct ggml_tensor * k = ggml_sqr(ctx, ggml_relu(ctx, ggml_mul_mat(ctx, layer.ffn_key, xk)));

            // r * (vw @ k)
            x = ggml_add_inplace(ctx, x, ggml_mul(ctx, r, ggml_mul_mat(ctx, layer.ffn_value, k)));
        }
    }

    // x = self.layer_norm(x, self.w.ln_out)
    x = rwkv_layer_norm(ctx, x, model.ln_out_weight, model.ln_out_bias);

    // x = (self.w.head.weight @ x).float()
    struct ggml_tensor * logits = ggml_mul_mat(ctx, model.head, x);

    ggml_build_forward_expand(cgraph.get(), logits);

    for (uint32_t i = 0; i < n_layer * 5; i++)
       ggml_build_forward_expand(cgraph.get(), state_parts[i]);

    out.state = state;
    out.state_parts = std::move(state_parts);
    out.token_index = token_index;
    out.logits = logits;
    out.cgraph = std::move(cgraph);
    return true;
}

struct rwkv_file_guard {
    FILE * file;
    ~rwkv_file_guard() { if (file) fclose(file); }
};

struct rwkv_ggml_guard {
    struct ggml_context * ctx;
    ~rwkv_ggml_guard() { if (ctx) ggml_free(ctx); }
};

void rwkv_set_print_errors(struct rwkv_context * ctx, bool print_errors) {
    bool * ptr = ctx ? &ctx->print_errors : &global_print_errors;
    *ptr = print_errors;
}

bool rwkv_get_print_errors(struct rwkv_context * ctx) {
    return ctx ? ctx->print_errors : global_print_errors;
}

enum rwkv_error_flags rwkv_get_last_error(struct rwkv_context * ctx) {
    enum rwkv_error_flags * ptr = ctx ? &ctx->last_error : &global_last_error;
    enum rwkv_error_flags value = *ptr;
    *ptr = RWKV_ERROR_NONE;
    return value;
}

struct rwkv_context * rwkv_init_from_file(const char * file_path, const uint32_t n_threads) {
    global_last_error = RWKV_ERROR_NONE;

    FILE * file = fopen(file_path, "rb");
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_OPEN, file, "failed to open file %s", file_path);
    rwkv_file_guard file_guard { file };

    struct stat64 file_stat;
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_STAT, fstat64(fileno(file), &file_stat) == 0, "failed to stat file %s", file_path);

    struct file_header header;
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_FILE, fread_file_header(file, header), "invalid file header");

    size_t memory_required = file_stat.st_size +
        // Intermediary vectors for calculation; there are around 100 calls to ggml
        size_t(100) * header.n_embed * sizeof(float) +
        // State, in and out
        size_t(2) * 5 * header.n_layer * header.n_embed * sizeof(float) +
        // Logits
        size_t(header.n_vocab) * sizeof(float) +
        // +256 MB just for any overhead
        // TODO This is too much for smaller models; need a more proper and robust way of measuring required memory
        size_t(256) * 1024 * 1024;

    struct ggml_context * ctx = ggml_init({ memory_required, NULL, false});
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL | RWKV_ERROR_ALLOC, ctx, "failed to create GGML context");
    rwkv_ggml_guard ggml_guard { ctx };

    std::unordered_map<std::string, struct ggml_tensor *> parameters;

    while ((size_t) ftell64(file) < (size_t) file_stat.st_size) {
        std::string name;
        struct ggml_tensor * tensor;
        RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS, fread_ggml_tensor(file, ctx, name, tensor), "failed to read model params");
        parameters[std::move(name)] = tensor;
    }

    file = NULL;
    file_guard = { NULL };

    struct rwkv_model model { header };

    std::unordered_map<std::string, struct ggml_tensor *> & parameters_ref = parameters;
    RWKV_ASSERT_NULL(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_PARAM_MISSING, rwkv_set_params(model, [&](const char * key, struct ggml_tensor *& dest) {
        struct ggml_tensor * tensor = parameters_ref[key];
        RWKV_ENSURE_OR_FALSE_MSG(tensor, "parameter %s not found", key);
        dest = tensor;
        return true;
    }));

    // Verify order of dimensions
    struct ggml_tensor * emb = model.emb;
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_SHAPE, emb->n_dims == 2, "unexpected dimension count of embedding matrix %d", emb->n_dims);
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, emb->ne[0] == header.n_embed, "unexpected dimension of embedding matrix %" PRId64, emb->ne[0]);
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, emb->ne[1] == header.n_vocab, "unexpected dimension of embedding matrix %" PRId64, emb->ne[1]);

    // Build graph
    struct rwkv_graph graph;
    RWKV_ASSERT_NULL(RWKV_ERROR_GRAPH, rwkv_build_graph(ctx, model, n_threads, graph));

    std::unique_ptr<struct rwkv_context> rwkv_ctx(new(std::nothrow) struct rwkv_context());
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, rwkv_ctx.get(), "failed to allocate context");

    ggml_guard.ctx = NULL; // don't free ggml context
    rwkv_ctx->model = std::move(model);
    rwkv_ctx->ctx = ctx;
    rwkv_ctx->graph = std::move(graph);
    rwkv_ctx->last_error = RWKV_ERROR_NONE;
    rwkv_ctx->print_errors = global_print_errors;

    return rwkv_ctx.release();
}

bool rwkv_eval(const struct rwkv_context * ctx, const uint32_t token, const float * state_in, float * state_out, float * logits_out) {
    ((struct rwkv_context *) ctx)->last_error = RWKV_ERROR_NONE;
    const struct file_header& header = ctx->model.header;
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, state_out != NULL, "state_out is NULL");
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, logits_out != NULL, "logits_out is NULL");
    RWKV_CTX_ASSERT_FALSE_MSG(ctx, RWKV_ERROR_ARGS, token < header.n_vocab, "Token is out of range 0..%d", header.n_vocab - 1);

    const struct rwkv_graph & graph = ctx->graph;

    ggml_set_i32_1d(graph.token_index, 0, token);

    if (state_in == NULL) {
        ggml_set_f32(graph.state, 0.0F);

        for (size_t i = 0; i < header.n_layer; i++) {
            // state[5 * i + 4] = -1e30
            ggml_set_f32(
                ggml_view_1d(ctx->ctx, graph.state, header.n_embed, (5 * i + 4) * header.n_embed * sizeof(float)),
                -1e30F
            );
        }
    } else {
        memcpy(graph.state->data, state_in, graph.state->ne[0] * sizeof(float));
    }

    ggml_graph_compute(ctx->ctx, graph.cgraph.get());

    for (size_t i = 0; i < header.n_layer * 5; i++) {
        struct ggml_tensor * part = graph.state_parts[i];
        memcpy(state_out + i * header.n_embed, part->data, part->ne[0] * sizeof(float));
    }

    memcpy(logits_out, graph.logits->data, graph.logits->ne[0] * sizeof(float));

    return true;
}

uint32_t rwkv_get_state_buffer_element_count(const struct rwkv_context * ctx) {
    return ctx->model.header.n_layer * 5 * ctx->model.header.n_embed;
}

uint32_t rwkv_get_logits_buffer_element_count(const struct rwkv_context * ctx) {
    return ctx->model.header.n_vocab;
}

void rwkv_free(struct rwkv_context * ctx) {
    std::unique_ptr<struct rwkv_context> rwkv_ctx(ctx);
    ggml_free(ctx->ctx);
}

bool rwkv_quantize_model_file(const char * input_path, const char * output_path, const char * target_name) {
    global_last_error = RWKV_ERROR_NONE;

    int32_t target_type = type_from_string(target_name);
    enum ggml_type target_ggml = type_to_ggml[target_type];

    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS | RWKV_ERROR_DATA_TYPE, target_type != TYPE_UNKNOWN, "invalid target data type (%s)", target_name);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS | RWKV_ERROR_DATA_TYPE, target_ggml != GGML_TYPE_UNKNOWN, "unsupported target data type (%s)", type_to_string[target_type]);

    RWKV_MSG("Loading model from '%s'\n", input_path);

    FILE * input = fopen(input_path, "rb");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_OPEN, input, "failed to open %s for reading", input_path);
    rwkv_file_guard input_guard { input };

    struct stat64 stat;
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_STAT, fstat64(fileno(input), &stat) == 0, "failed to stat file %s", input_path);

    FILE * output = fopen(output_path, "wb");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_OPEN, output, "failed to open %s for writing", output_path);
    rwkv_file_guard output_guard { output };

    struct file_header file_header;
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE, fread_file_header(input, file_header), "invalid file header");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE, fwrite_file_header(output, file_header), "failed to write file header");
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE, file_header.data_type == TYPE_F32 || file_header.data_type == TYPE_F16, "unsupported source data type (%s); needs to be f32 or f16", type_to_string[file_header.data_type]);

    // required to init the fp16 tables
    // doesn't crash if ggml_init fails
    ggml_free(ggml_init({ 0, NULL, true }));

    // Process parameters
    size_t orig_total_size = 0;
    size_t new_total_size = 0;

    int64_t hist_all[16] {};

    std::vector<uint8_t> a;
    std::vector<uint8_t> b;

    std::vector<uint8_t> * container = &a;
    std::vector<uint8_t> * scratch = &b;

    while (ftell(input) < stat.st_size) {
        struct tensor_header header;
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS, fread_tensor_header(input, header), "invalid tensor header");

        std::string name((size_t) header.key_length, '\0');
        const char * name_str = name.c_str();
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_KEY, fread_tensor_key(input, header, name.c_str()), "failed to read tensor name");

        container->resize(tensor_bytes(header));
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_DATA, fread_tensor_data(input, header, container->data()), "failed to read tensor data of %s", name_str);

        RWKV_MSG("%48s - [%5" PRId32 ", %5" PRId32 "], type = %6s ", name_str, header.width, header.height, type_to_string[header.data_type]);
        
        int32_t orig_type = header.data_type;
        size_t orig_size = tensor_bytes(header);
        size_t new_size = orig_size;

        if ((orig_type == TYPE_F32 || orig_type == TYPE_F16) && header.dim_count == 2 && name != "emb.weight" && name != "head.weight") {
            RWKV_MSG("quantizing... ");

            size_t nelements = (size_t) header.width * (size_t) header.height;

            if (orig_type == TYPE_F16) {
                header.data_type = TYPE_F32;
                scratch->resize(new_size = tensor_bytes(header));
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) container->data(), (float *) scratch->data(), nelements);
                std::swap(container, scratch);
            }

            header.data_type = target_type;
            scratch->resize(new_size = tensor_bytes(header));

            int64_t hist_cur[16] {};
            new_size = ggml_quantize_chunk(target_ggml, (const float *) container->data(), scratch->data(), 0, nelements, hist_cur);
            std::swap(container, scratch);

            orig_total_size += orig_size;
            new_total_size += new_size;

            RWKV_MSG("size = %8.2f MB -> %8.2f MB | hist: ", orig_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);

            for (int i = 0; i < 16; i++) {
                RWKV_MSG("%5.3f ", hist_cur[i] / (float) nelements);
                hist_all[i] += hist_cur[i];
            }

            RWKV_MSG("\n");
        } else {

            RWKV_MSG("size = %8.3f MB\n", orig_size / 1024.0 / 1024.0);
            orig_total_size += orig_size;
            new_total_size += orig_size;
        }

        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_WRITE, fwrite_tensor_header(output, header), "failed to write tensor header of %s", name_str);
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_WRITE, fwrite_string(output, name), "failed to write tensor name of %s", name_str);
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_WRITE, fwrite_data(output, container->data(), new_size), "failed to write tensor data of %s", name_str);
    }

    RWKV_MSG("original size     = %8.2f MB\n", orig_total_size / 1024.0 / 1024.0);
    RWKV_MSG("quantized size    = %8.2f MB\n", new_total_size / 1024.0 / 1024.0);
    RWKV_MSG("compression ratio = %8.2f\n", orig_total_size / float(new_total_size));

    int64_t sum_all = 0;

    for (int i = 0; i < 16; i++)
        sum_all += hist_all[i];

    RWKV_MSG("hist: ");

    for (int i = 0; i < 16; ++i)
        printf("%5.3f ", hist_all[i] / float(sum_all));

    RWKV_MSG("\n");

    return true;
}

const char * rwkv_get_system_info_string(void) {
    static std::string s;

    s  = "";
    s += "AVX="       + std::to_string(ggml_cpu_has_avx())       + ", ";
    s += "AVX2="      + std::to_string(ggml_cpu_has_avx2())      + ", ";
    s += "AVX512="    + std::to_string(ggml_cpu_has_avx512())    + ", ";
    s += "FMA="       + std::to_string(ggml_cpu_has_fma())       + ", ";
    s += "NEON="      + std::to_string(ggml_cpu_has_neon())      + ", ";
    s += "ARM_FMA="   + std::to_string(ggml_cpu_has_arm_fma())   + ", ";
    s += "F16C="      + std::to_string(ggml_cpu_has_f16c())      + ", ";
    s += "FP16_VA="   + std::to_string(ggml_cpu_has_fp16_va())   + ", ";
    s += "WASM_SIMD=" + std::to_string(ggml_cpu_has_wasm_simd()) + ", ";
    s += "BLAS="      + std::to_string(ggml_cpu_has_blas())      + ", ";
    s += "SSE3="      + std::to_string(ggml_cpu_has_sse3())      + ", ";
    s += "VSX="       + std::to_string(ggml_cpu_has_vsx());

    return s.c_str();
}