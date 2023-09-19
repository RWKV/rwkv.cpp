#include "rwkv.h"
#include "ggml.h"
#include "ggml-alloc.h"

#ifdef GGML_USE_CUBLAS
#include "ggml/src/ggml-cuda.h"
#elif defined(GGML_USE_CLBLAST)
#include "ggml/src/ggml-opencl.h"
#endif

#include <string>
#include <vector>
#include <cstring>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <memory>
#include <utility>

#define _FILE_OFFSET_BITS 64
// Puts an optional break point, if debug is enabled.
#define RWKV_MAYBE_BREAK

#include <sys/stat.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define stat _stat64
#define fstat _fstat64
#define ftell _ftelli64
#define fseek _fseeki64

#ifndef NDEBUG
#include <intrin.h>
#define RWKV_MAYBE_BREAK __debugbreak()
#endif
#else
#if !defined(__APPLE__)
#define ftell ftello
#define fseek fseeko
#endif
#endif

static_assert(sizeof(stat::st_size) >= 8, "File offsets should be 64-bit or else rwkv.cpp will not be able to load model files over 2 GB");
static_assert(sizeof(decltype(ftell(NULL))) >= 8, "File offsets should be 64-bit or else rwkv.cpp will not be able to load model files over 2 GB");

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
#define RWKV_CTX_MSG(ctx, ...) do { if (ctx->print_errors) fprintf(stderr, __VA_ARGS__); } while (0)

// If the condition x is false, adds ERR_VAL to the last error, and returns RET_VAL.
#define RWKV_ASSERT(ERR_VAL, RET_VAL, x) do { \
    if (!(x)) { \
        global_last_error |= ERR_VAL; \
        RWKV_MSG("\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

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
#define RWKV_CTX_ASSERT(ctx, ERR_VAL, RET_VAL, x) do { \
    if (!(x)) { \
        ((struct rwkv_context *) ctx)->last_error |= ERR_VAL; \
        RWKV_CTX_MSG(ctx, "\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

// If the condition x is false, returns RET_VAL.
#define RWKV_ENSURE(RET_VAL, x) do { \
    if (!(x)) { \
        RWKV_MSG("\n%s:%d: %s\n", __FILE__, __LINE__, #x); \
        RWKV_MAYBE_BREAK; \
        return RET_VAL; \
    } } while (0)

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

#define RWKV_ASSERT_FALSE(ERR_VAL, x) RWKV_ASSERT(ERR_VAL, false, x)
#define RWKV_ASSERT_NULL(ERR_VAL, x) RWKV_ASSERT(ERR_VAL, NULL, x)

#define RWKV_CTX_ASSERT_FALSE(ctx, ERR_VAL, x) RWKV_CTX_ASSERT(ctx, ERR_VAL, false, x)

#define RWKV_ENSURE_OR_FALSE(x) RWKV_ENSURE(false, x)
#define RWKV_ENSURE_OR_NULL(x) RWKV_ENSURE(NULL, x)
#define RWKV_ENSURE_OR_FALSE_MSG(x, ...) RWKV_ENSURE_MSG(false, x, __VA_ARGS__)

// --- Utilities ---

size_t rwkv_tensor_nbytes(const enum ggml_type type, const int64_t width, const int64_t height) {
    return (ggml_type_size(type) * width * height) / ggml_blck_size(type);
}

// For some reason, ggml_nbytes calculates the size in a way incompatible with rwkv.cpp
size_t rwkv_tensor_nbytes(const struct ggml_tensor * tensor) {
    return rwkv_tensor_nbytes(tensor->type, tensor->ne[0], tensor->ne[1]);
}

size_t rwkv_ggml_overhead() {
    return ggml_tensor_overhead() * GGML_MAX_NODES + ggml_graph_overhead();
}

struct ggml_context * rwkv_init_ggml_context(const size_t memory_size, const bool no_alloc) {
    struct ggml_init_params init_params = {
        memory_size,
        NULL,
        no_alloc
    };

    return ggml_init(init_params);
}

// --- IO utilities ---

// Reads a single uint32 value from a file.
bool rwkv_fread_uint32(FILE * file, uint32_t & dest) {
    return fread((void *) &dest, sizeof(uint32_t), 1, file) == 1;
}

// Reads a single string value from a file.
bool rwkv_fread_string(FILE * file, size_t length, std::string & dest) {
    dest.resize(length);
    return fread((void *) dest.data(), length, 1, file) == 1;
}

// Reads a single data buffer from a file.
bool rwkv_fread_data(FILE * file, size_t length, void * dest) {
    return fread(dest, length, 1, file) == 1;
}

// Writes a single uint32 value to a file.
bool rwkv_fwrite_uint32(FILE * file, const uint32_t value) {
    return fwrite((const void *) &value, sizeof(uint32_t), 1, file);
}

// Writes a single string value to a file.
bool rwkv_fwrite_string(FILE * file, const std::string & value) {
    return fwrite((const void *) value.data(), value.length(), 1, file) == 1;
}

// Writes a single data buffer to a file.
bool rwkv_fwrite_data(FILE * file, const void * data, const size_t length) {
    return fwrite(data, length, 1, file) == 1;
}

// --- File handling ---

#define TYPE_UNKNOWN TYPE_COUNT

enum rwkv_type {
    TYPE_FP32,
    TYPE_FP16,
    TYPE_Q4_0,
    TYPE_Q4_1,
    TYPE_Q4_1_O, // Unsupported
    TYPE_Q4_2, // Unsupported
    TYPE_Q4_3, // Unsupported
    TYPE_Q5_0,
    TYPE_Q5_1,
    TYPE_Q8_0,
    TYPE_COUNT
};

#define GGML_TYPE_UNKNOWN GGML_TYPE_COUNT

extern const enum ggml_type rwkv_type_to_ggml[TYPE_COUNT + 1] = {
    GGML_TYPE_F32,     /* FP32   */
    GGML_TYPE_F16,     /* FP16   */
    GGML_TYPE_Q4_0,    /* Q4_0   */
    GGML_TYPE_Q4_1,    /* Q4_1   */
    GGML_TYPE_UNKNOWN, /* Q4_1_O */
    GGML_TYPE_UNKNOWN, /* Q4_2   */
    GGML_TYPE_UNKNOWN, /* Q4_3   */
    GGML_TYPE_Q5_0,    /* Q5_0   */
    GGML_TYPE_Q5_1,    /* Q5_1   */
    GGML_TYPE_Q8_0,    /* Q8_0   */
    GGML_TYPE_COUNT    /* COUNT  */
};

extern const enum rwkv_type rwkv_type_from_ggml[GGML_TYPE_COUNT + 1] = {
    TYPE_FP32,   /* FP32  */
    TYPE_FP16,   /* FP16  */
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

extern const char * rwkv_type_to_string[TYPE_COUNT + 1] = {"FP32", "FP16", "Q4_0", "Q4_1", "Q4_1_O", "Q4_2", "Q4_3", "Q5_0", "Q5_1", "Q8_0", "unknown"};

enum rwkv_type rwkv_type_from_string(const char * str) {
    for (int ord = 0; ord < TYPE_COUNT; ord++) {
        if (strcmp(str, rwkv_type_to_string[ord]) == 0) {
            return (enum rwkv_type) ord;
        }
    }

    return TYPE_UNKNOWN;
}

struct rwkv_file_header {
    uint32_t magic;
    uint32_t version;
    uint32_t n_vocab;
    uint32_t n_embed;
    uint32_t n_layer;
    uint32_t data_type;
};

bool rwkv_is_file_version_in_range(uint32_t version) {
    return version >= RWKV_FILE_VERSION_MIN && version <= RWKV_FILE_VERSION_MAX;
}

bool rwkv_fread_file_header(FILE * file, struct rwkv_file_header & header, bool verify_data_type = true) {
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, rwkv_fread_data(file, sizeof(struct rwkv_file_header), &header));
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_MAGIC, header.magic == RWKV_FILE_MAGIC);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_VERSION, rwkv_is_file_version_in_range(header.version), "Unsupported file version %" PRId32, header.version);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_DATA_TYPE, header.data_type < TYPE_COUNT, "Model data type out of range (%" PRId32 " > %" PRId32 ")", header.data_type, TYPE_COUNT - 1);

    if (verify_data_type) {
        enum ggml_type ggml_type = rwkv_type_to_ggml[header.data_type];

        RWKV_ASSERT_FALSE_MSG(
            RWKV_ERROR_DATA_TYPE,
            ggml_type != GGML_TYPE_UNKNOWN,
            "Models in %s format cannot be loaded anymore because the format was removed.\n"
            "You need to quantize the model into another format or use an older version of rwkv.cpp.\n"
            "See https://github.com/saharNooby/rwkv.cpp#compatibility for more info",
            rwkv_type_to_string[header.data_type]
        );

        RWKV_ASSERT_FALSE_MSG(
            RWKV_ERROR_DATA_TYPE,
            (!ggml_is_quantized(ggml_type) || header.version == RWKV_FILE_VERSION_1),
            "The quantized model file in %s format was created with an old version of rwkv.cpp and can not be loaded anymore.\n"
            "You need to requantize the model or use an older version of rwkv.cpp.\n"
            "See https://github.com/saharNooby/rwkv.cpp#compatibility for more info",
            rwkv_type_to_string[header.data_type]
        );
    }

    return true;
}

bool rwkv_fwrite_file_header(FILE * file, const struct rwkv_file_header & header) {
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, rwkv_fwrite_data(file, &header, sizeof(struct rwkv_file_header)));
    return true;
}

struct rwkv_tensor_header {
    uint32_t dim_count;
    uint32_t key_length;
    uint32_t data_type;
    uint32_t width;
    uint32_t height;

    size_t size() const;
};

size_t rwkv_tensor_header::size() const {
    return rwkv_tensor_nbytes(rwkv_type_to_ggml[this->data_type], this->width, this->height);
}

struct rwkv_tensor {
    struct rwkv_tensor_header header;
    std::string name;
    uint8_t * data;
};

bool rwkv_fread_tensor_header(FILE * file, struct rwkv_tensor_header & header) {
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, rwkv_fread_data(file, sizeof(struct rwkv_tensor_header) - sizeof(uint32_t), &header));
    header.height = 1;
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_SHAPE, header.dim_count == 1 || header.dim_count == 2, "Tensor has an invalid shape (%" PRId32 " dimensions)", header.dim_count);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_DATA_TYPE, header.data_type < TYPE_COUNT, "Tensor data type out of range (%" PRId32 " > %" PRId32 ")", header.data_type, TYPE_COUNT - 1);
    RWKV_ASSERT_FALSE_MSG(
        RWKV_ERROR_DATA_TYPE,
        rwkv_type_to_ggml[header.data_type] != GGML_TYPE_UNKNOWN,
        "Tensor data type (%s) is no longer supported",
        rwkv_type_to_string[header.data_type]
    );

    if (header.dim_count == 2) {
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, rwkv_fread_uint32(file, header.height));
    }

    return true;
}

bool rwkv_fwrite_tensor_header(FILE * file, const struct rwkv_tensor_header & header) {
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_WRITE, rwkv_fwrite_data(file, &header, sizeof(struct rwkv_tensor_header) - (header.dim_count == 1 ? sizeof(uint32_t) : 0)));
    return true;
}

bool rwkv_fskip_tensor_name_and_data(FILE * file, const struct rwkv_tensor_header & header) {
    return fseek(file, header.key_length + header.size(), SEEK_CUR) == 0;
}

bool rwkv_fskip_tensor_data(FILE * file, const struct rwkv_tensor_header & header) {
    return fseek(file, header.size(), SEEK_CUR) == 0;
}

bool rwkv_fread_tensor_header_and_skip(FILE * file, struct rwkv_tensor_header & header) {
    RWKV_ENSURE_OR_FALSE(rwkv_fread_tensor_header(file, header));
    RWKV_ASSERT_FALSE(RWKV_ERROR_DATA, rwkv_fskip_tensor_name_and_data(file, header));
    return true;
}

bool rwkv_fread_tensor_data(FILE * file, struct rwkv_tensor & output, void * buffer = NULL) {
    size_t data_size = output.header.size();
    RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, rwkv_fread_string(file, output.header.key_length, output.name));

    if (buffer) {
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, rwkv_fread_data(file, data_size, buffer));
    } else {
        output.data = NULL;
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE_READ, rwkv_fskip_tensor_name_and_data(file, output.header));
    }

    return true;
}

bool rwkv_fread_tensor(FILE * file, struct rwkv_tensor & output, void * buffer = NULL) {
    RWKV_ENSURE_OR_FALSE(rwkv_fread_tensor_header(file, output.header));
    RWKV_ENSURE_OR_FALSE(rwkv_fread_tensor_data(file, output, buffer));
    return true;
}

bool rwkv_fread_ggml_tensor_data(FILE * file, const struct rwkv_tensor_header & header, struct ggml_context * ctx, std::string & name, struct ggml_tensor *& tensor) {
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_READ, rwkv_fread_string(file, header.key_length, name), "Failed to read tensor name");

    enum ggml_type ggml_type = rwkv_type_to_ggml[header.data_type];
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_UNSUPPORTED, ggml_type != GGML_TYPE_UNKNOWN, "Unsupported tensor data type %s from %s", rwkv_type_to_string[header.data_type], name.c_str());

    tensor = header.dim_count == 1
        ? ggml_new_tensor_1d(ctx, ggml_type, header.width)
        : ggml_new_tensor_2d(ctx, ggml_type, header.width, header.height);

    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, tensor, "Failed to allocate tensor");
    ggml_set_name(tensor, name.c_str());

    // Tensor data may be NULL if no_alloc is true
    if (tensor->data != NULL) {
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_READ, rwkv_fread_data(file, rwkv_tensor_nbytes(tensor), tensor->data), "Failed to read tensor data from %s", name.c_str());
    } else {
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_READ, rwkv_fskip_tensor_data(file, header), "Failed to skip tensor data from %s", name.c_str());
    }

    return true;
}

bool rwkv_fread_ggml_tensor(FILE * file, struct ggml_context * ctx, std::string & name, struct ggml_tensor *& tensor) {
    struct rwkv_tensor_header header;
    RWKV_ENSURE_OR_FALSE_MSG(rwkv_fread_tensor_header(file, header), "Invalid tensor header");
    return rwkv_fread_ggml_tensor_data(file, header, ctx, name, tensor);
}

bool rwkv_fwrite_tensor(FILE * file, const struct rwkv_tensor & tensor) {
    RWKV_ENSURE_OR_FALSE(rwkv_fwrite_tensor_header(file, tensor.header));
    RWKV_ENSURE_OR_FALSE(rwkv_fwrite_string(file, tensor.name));
    RWKV_ENSURE_OR_FALSE(rwkv_fwrite_data(file, tensor.data, tensor.header.size()));
    return true;
}

// --- Model loading ---

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
bool rwkv_set_params(struct rwkv_model & model, F callback) {
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
bool rwkv_load_model_from_file(const char * file_path, struct rwkv_model & model) {
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

    // Verify order of dimensions
    struct ggml_tensor * emb = model.emb;
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_SHAPE, emb->n_dims == 2, "Unexpected dimension count of embedding matrix %d", emb->n_dims);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, emb->ne[0] == model.header.n_embed, "Unexpected dimension of embedding matrix %" PRId64, emb->ne[0]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS | RWKV_ERROR_DIMENSION, emb->ne[1] == model.header.n_vocab, "Unexpected dimension of embedding matrix %" PRId64, emb->ne[1]);

    return true;
}

// --- Operators ---

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

// --- Implementation ---

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

void rwkv_carry_x(struct ggml_context * ctx,
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

void rwkv_att_rkv(
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

struct ggml_tensor * rwkv_att_wkv(
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

struct ggml_tensor * rwkv_att(struct ggml_context * ctx, struct ggml_tensor * x, struct rwkv_layer layer, struct rwkv_layer_state & state) {
    struct ggml_tensor * x_prev;
    rwkv_carry_x(ctx, layer.ln1_weight, layer.ln1_bias, x, x_prev, state.att_xx);

    struct ggml_tensor * r, * k, * v;
    rwkv_att_rkv(ctx, layer, x, x_prev, r, k, v);

    struct ggml_tensor * wkv = rwkv_att_wkv(ctx, layer.att_time_first, layer.att_time_decay, k, v, state.att_aa, state.att_bb, state.att_pp);

    // ow @ (r * xx)
    return ggml_mul_mat(ctx, layer.att_output, ggml_mul(ctx, r, wkv));
}

struct ggml_tensor * rwkv_ffn(struct ggml_context * ctx, struct ggml_tensor * x, struct rwkv_layer layer, struct rwkv_layer_state & state) {
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

void rwkv_create_input_and_output_views(
    struct rwkv_layer_state * inputs,
    struct rwkv_layer_state * outputs,
    struct ggml_tensor * input,
    struct ggml_tensor * output,
    struct ggml_context * ctx,
    size_t n_layer,
    size_t n_embed
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

// Creates and sets the input and output ggml tensors, builds the computation graph.
bool rwkv_build_serial_graph(struct rwkv_model & model, struct rwkv_computation_graph & graph) {
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

    rwkv_create_input_and_output_views(inputs.get(), outputs.get(), input, output, ctx, n_layer, n_embed);

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

// Stolen from llama.cpp.
static const size_t tensor_alignment = 32;

// Prepares the computation graph for inference, measuring and allocating all input and output tensors.
bool rwkv_measure_and_build_serial_context(struct rwkv_model & model, struct rwkv_computation_graph & graph) {
    if (graph.ggml_ctx) {
        ggml_free(graph.ggml_ctx);

        graph.ggml_ctx = NULL;
    }

    // 1. Measure the space required for the ggml context.
    graph.ggml_ctx = rwkv_init_ggml_context(rwkv_ggml_overhead(), true);

    RWKV_ENSURE_OR_FALSE(rwkv_build_serial_graph(model, graph));

    struct ggml_allocr * allocator = ggml_allocr_new_measure(tensor_alignment);

    size_t required_context_size = ggml_allocr_alloc_graph(allocator, graph.cgraph.get()) +
            + rwkv_ggml_overhead()
            + tensor_alignment
            // For some reason, calculation above does not result in enough memory allocated.
            // Instead of diving deep into ggml internals to debug this issue, I will just add some padding.
            // 64 MB seems to be enough for Raven 14B model.
            + size_t(64) * 1024 * 1024;

    ggml_allocr_free(allocator);
    ggml_free(graph.ggml_ctx);

    // 2. Create the real ggml context.
    graph.ggml_ctx = rwkv_init_ggml_context(required_context_size, false);

    RWKV_ENSURE_OR_FALSE(rwkv_build_serial_graph(model, graph));

    return true;
}

// ---

// Creates and sets the input and output ggml tensors, builds the computation graph.
bool rwkv_build_sequential_graph(struct rwkv_model & model, struct rwkv_computation_graph & graph, const size_t sequence_length) {
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

    rwkv_create_input_and_output_views(inputs.get(), outputs.get(), input, output, ctx, n_layer, n_embed);

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
bool rwkv_measure_and_build_sequential_context(struct rwkv_model & model, struct rwkv_computation_graph & graph, const size_t sequence_length) {
    if (graph.ggml_ctx) {
        ggml_free(graph.ggml_ctx);

        graph.ggml_ctx = NULL;
    }

    // 1. Measure the space required for the ggml context.
    graph.ggml_ctx = rwkv_init_ggml_context(rwkv_ggml_overhead(), true);

    RWKV_ENSURE_OR_FALSE(rwkv_build_sequential_graph(model, graph, sequence_length));

    struct ggml_allocr * allocator = ggml_allocr_new_measure(tensor_alignment);

    size_t required_context_size = ggml_allocr_alloc_graph(allocator, graph.cgraph.get()) +
            + rwkv_ggml_overhead()
            + tensor_alignment
            // For some reason, calculation above does not result in enough memory allocated.
            // Instead of diving deep into ggml internals to debug this issue, I will just add some padding.
            // 64 MB per token seems to be enough for Raven 14B model. It works for sequence_length = 5; not tested on larger lengths.
            + sequence_length * 64 * 1024 * 1024;

    ggml_allocr_free(allocator);
    ggml_free(graph.ggml_ctx);

    // 2. Create the real ggml context.
    graph.ggml_ctx = rwkv_init_ggml_context(required_context_size, false);

    RWKV_ENSURE_OR_FALSE(rwkv_build_sequential_graph(model, graph, sequence_length));

    return true;
}

// ---

struct rwkv_context * rwkv_init_from_file(const char * file_path, const uint32_t n_threads) {
    global_last_error = RWKV_ERROR_NONE;

    std::unique_ptr<struct rwkv_context> ctx(new(std::nothrow) struct rwkv_context());
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, ctx, "Failed to allocate rwkv_context");

    ctx->model = new(std::nothrow) struct rwkv_model();
    ctx->model->reference_count++;
    RWKV_ENSURE_OR_NULL(rwkv_load_model_from_file(file_path, *ctx->model));

    ctx->n_threads = n_threads;

    RWKV_ENSURE_OR_NULL(rwkv_measure_and_build_serial_context(*ctx->model, ctx->serial_graph));

    return ctx.release();
}

struct rwkv_context * rwkv_clone_context(struct rwkv_context * ctx, const uint32_t n_threads) {
    std::unique_ptr<struct rwkv_context> clone(new(std::nothrow) struct rwkv_context());
    RWKV_ASSERT_NULL_MSG(RWKV_ERROR_CTX | RWKV_ERROR_ALLOC, clone, "Failed to allocate rwkv_context");

    clone->model = ctx->model;
    clone->model->reference_count++;

    clone->n_threads = n_threads;

    RWKV_ENSURE_OR_NULL(rwkv_measure_and_build_serial_context(*clone->model, clone->serial_graph));

    clone->last_used_sequence_length = 0;

    clone->print_errors = ctx->print_errors;

    return clone.release();
}

bool rwkv_gpu_offload_layers(struct rwkv_context * ctx, const uint32_t n_layers) {
#if defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
    const auto offload = [&](struct ggml_tensor * tensor) {
        // TODO Support multi-GPU
        tensor->backend = GGML_BACKEND_GPU;
#ifdef GGML_USE_CUBLAS
        ggml_cuda_transform_tensor(tensor->data, tensor);
#elif defined(GGML_USE_CLBLAST)
        ggml_cl_transform_tensor(tensor->data, tensor);
#endif
    };

    const size_t n_gpu = std::min(n_layers, ctx->model->header.n_layer);

    if (ctx->model->offloaded_layer_count < n_gpu) {
        for (size_t & i = ctx->model->offloaded_layer_count; i < n_gpu; i++) {
            const struct rwkv_layer & layer = ctx->model->layers[i];

            // TODO Also offload other operations to GPU with ggml_cuda_assign_buffers
            offload(layer.att_key);
            offload(layer.att_value);
            offload(layer.att_receptance);
            offload(layer.att_output);

            offload(layer.ffn_key);
            offload(layer.ffn_value);
            offload(layer.ffn_receptance);
        }

        return true;
    }
#endif
    return false;
}

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

// Provided for compatibility.
extern "C" RWKV_API uint32_t rwkv_get_state_buffer_element_count(const struct rwkv_context * ctx) {
    return rwkv_get_state_len(ctx);
}

// Provided for compatibility.
extern "C" RWKV_API uint32_t rwkv_get_logits_buffer_element_count(const struct rwkv_context * ctx) {
    return rwkv_get_logits_len(ctx);
}

extern "C" RWKV_API size_t rwkv_get_n_vocab(const struct rwkv_context * ctx) {
    return (size_t) ctx->model->header.n_vocab;
}

extern "C" RWKV_API size_t rwkv_get_n_embed(const struct rwkv_context * ctx) {
    return (size_t) ctx->model->header.n_embed;
}

extern "C" RWKV_API size_t rwkv_get_n_layer(const struct rwkv_context * ctx) {
    return (size_t) ctx->model->header.n_layer;
}

size_t rwkv_get_state_len(const struct rwkv_context * ctx) {
    const struct rwkv_file_header & header = ctx->model->header;

    return (size_t) header.n_embed * 5 * (size_t) header.n_layer;
}

size_t rwkv_get_logits_len(const struct rwkv_context * ctx) {
    return (size_t) ctx->model->header.n_vocab;
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

void rwkv_free(struct rwkv_context * ctx) {
    if (--ctx->model->reference_count == 0) {
        ggml_free(ctx->model->ggml_ctx);

        delete ctx->model;
    }

    ggml_free(ctx->serial_graph.ggml_ctx);

    if (ctx->last_used_sequence_length > 0) {
        ggml_free(ctx->sequential_graph.ggml_ctx);
    }

    std::unique_ptr<struct rwkv_context> rwkv_ctx(ctx);
}

bool rwkv_quantize_model_file(const char * in_path, const char * out_path, const char * type_name) {
    global_last_error = RWKV_ERROR_NONE;

    enum ggml_type out_type = rwkv_type_to_ggml[rwkv_type_from_string(type_name)];
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ARGS | RWKV_ERROR_DATA_TYPE, ggml_is_quantized(out_type), "Unsupported output data type (%s)", rwkv_type_to_string[rwkv_type_from_ggml[out_type]]);

    RWKV_MSG("Loading model from '%s'\n", in_path);

    struct stat in_stat;

    struct rwkv_file in_file(fopen(in_path, "rb"));
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_OPEN, in_file.file, "Failed to open %s for reading", in_path);

    // Be very careful when changing this code. It must support files larger than 2 GB by using 64-bit functions to the get file length.
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_STAT, fstat(fileno(in_file.file), &in_stat) == 0, "failed to stat file %s", in_path);

    struct rwkv_file out_file(fopen(out_path, "wb"));
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_OPEN, out_file.file, "Failed to open %s for writing", out_path);

    struct rwkv_file_header in_header;
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE, rwkv_fread_file_header(in_file.file, in_header), "Invalid file header");

    enum ggml_type in_type = rwkv_type_to_ggml[in_header.data_type];
    RWKV_ASSERT_FALSE_MSG(
        RWKV_ERROR_FILE,
        in_type == GGML_TYPE_F32 || in_type == GGML_TYPE_F16,
        "Unsupported input data type (%s); needs to be FP32 or FP16",
        rwkv_type_to_string[rwkv_type_from_ggml[in_type]]
    );

    struct rwkv_file_header out_header = in_header;
    out_header.version = RWKV_FILE_VERSION;
    out_header.data_type = rwkv_type_from_ggml[out_type];
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE, rwkv_fwrite_file_header(out_file.file, out_header), "Failed to write file header");

    // Process parameters
    size_t orig_total_size = 0;
    size_t new_total_size = 0;

    // Required to init the F16 tables
    // Doesn't crash if ggml_init fails
    ggml_free(ggml_init({ 0, NULL, true }));

    size_t max_in_size = 0;
    size_t max_out_size = 0;
    size_t max_key_length = 0;

    while (ftell(in_file.file) < in_stat.st_size) {
        struct rwkv_tensor_header header;
        RWKV_ASSERT_FALSE(RWKV_ERROR_FILE, rwkv_fread_tensor_header_and_skip(in_file.file, header));

        size_t in_size = header.size();

        if (in_size > max_in_size) {
            max_in_size = in_size;
        }

        // f16 type tensors get relocated to out and then converted into f32 at in
        if (header.data_type == TYPE_FP16) {
            if (in_size > max_out_size) {
                max_out_size = in_size;
            }

            size_t f32_size = rwkv_tensor_nbytes(GGML_TYPE_F32, header.width, header.height);

            if (f32_size > max_in_size) {
                max_in_size = f32_size;
            }
        }

        size_t out_size = rwkv_tensor_nbytes(out_type, header.width, header.height);

        if (out_size > max_out_size) {
            max_out_size = out_size;
        }

        if (header.key_length > max_key_length) {
            max_key_length = header.key_length;
        }
    }

    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE | RWKV_ERROR_FILE_READ, fseek(in_file.file, sizeof(struct rwkv_file_header), SEEK_SET) == 0, "Failed to seek in file");

    // This is a histogram of quantized values. If it shows single 1.0, then all 0.0, something went very wrong!
    int64_t hist_all[16] {};

    std::unique_ptr<uint8_t[]> scratch(new(std::nothrow) uint8_t[max_in_size + max_out_size]);
    RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_ALLOC, scratch.get(), "Failed to allocate buffer");

    uint8_t * in_buf = scratch.get();
    uint8_t * out_buf = in_buf + max_in_size;

    struct rwkv_tensor tensor;
    struct rwkv_tensor_header & header = tensor.header;
    std::string & name = tensor.name;
    uint8_t *& data = tensor.data;

    while (ftell(in_file.file) < in_stat.st_size) {
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS, rwkv_fread_tensor_header(in_file.file, header), "Failed to read tensor header");
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS, rwkv_fread_string(in_file.file, header.key_length, name), "Failed to read tensor name");

        const char * name_str = name.c_str();
        RWKV_MSG("%*s - [%5" PRId32 ", %5" PRId32 "], type = %6s ", (int) max_key_length, name_str, header.width, header.height, rwkv_type_to_string[header.data_type]);

        data = header.data_type == TYPE_FP16 ? out_buf : in_buf;
        size_t orig_size = header.size(), new_size = orig_size;
        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_MODEL_PARAMS, rwkv_fread_data(in_file.file, orig_size, data), "\nFailed to read tensor data of %s", name_str);

        // Quantize only 2D tensors, except embedding and head matrices.
        // Embedding and head take not too much space, especially in bigger models;
        // but they significantly increase perplexity when quantized.
        if ((header.data_type == TYPE_FP32 || header.data_type == TYPE_FP16) && header.dim_count == 2 && name != "emb.weight" && name != "head.weight") {
            RWKV_MSG("quantizing... ");

            size_t nelements = (size_t) header.width * (size_t) header.height;

            if (header.data_type == TYPE_FP16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) out_buf, (float *) in_buf, nelements);
            }

            int64_t hist_cur[16] {};
            new_size = ggml_quantize_chunk(out_type, (const float *) in_buf, out_buf, 0, nelements, hist_cur);
            header.data_type = rwkv_type_from_ggml[out_type];
            data = out_buf;

            RWKV_MSG("size = %8.2f MB -> %8.2f MB | hist: ", orig_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);

            for (int i = 0; i < 16; i++) {
                RWKV_MSG("%5.3f ", hist_cur[i] / (float) nelements);
                hist_all[i] += hist_cur[i];
            }

            RWKV_MSG("\n");
        } else {
            RWKV_MSG("size = %8.3f MB\n", orig_size / 1024.0 / 1024.0);
        }

        RWKV_ASSERT_FALSE_MSG(RWKV_ERROR_FILE_WRITE, rwkv_fwrite_tensor(out_file.file, tensor), "Failed to write tensor %s", name_str);
        orig_total_size += orig_size;
        new_total_size += new_size;
    }

    RWKV_MSG("original size     = %8.2f MB\n", orig_total_size / 1024.0 / 1024.0);
    RWKV_MSG("quantized size    = %8.2f MB\n", new_total_size / 1024.0 / 1024.0);
    RWKV_MSG("compression ratio = %8.2f\n", orig_total_size / float(new_total_size));

    int64_t sum_all = 0;

    for (int i = 0; i < 16; i++) {
        sum_all += hist_all[i];
    }

    RWKV_MSG("hist: ");

    for (int i = 0; i < 16; ++i) {
        printf("%5.3f ", hist_all[i] / float(sum_all));
    }

    RWKV_MSG("\n");

    return true;
}

const char * rwkv_get_system_info_string(void) {
    static std::string s;

    if (s.empty()) {
        s  = "";
        s += "AVX="       + std::to_string(ggml_cpu_has_avx())       + " ";
        s += "AVX2="      + std::to_string(ggml_cpu_has_avx2())      + " ";
        s += "AVX512="    + std::to_string(ggml_cpu_has_avx512())    + " ";
        s += "FMA="       + std::to_string(ggml_cpu_has_fma())       + " ";
        s += "NEON="      + std::to_string(ggml_cpu_has_neon())      + " ";
        s += "ARM_FMA="   + std::to_string(ggml_cpu_has_arm_fma())   + " ";
        s += "F16C="      + std::to_string(ggml_cpu_has_f16c())      + " ";
        s += "FP16_VA="   + std::to_string(ggml_cpu_has_fp16_va())   + " ";
        s += "WASM_SIMD=" + std::to_string(ggml_cpu_has_wasm_simd()) + " ";
        s += "BLAS="      + std::to_string(ggml_cpu_has_blas())      + " ";
        s += "SSE3="      + std::to_string(ggml_cpu_has_sse3())      + " ";
        s += "VSX="       + std::to_string(ggml_cpu_has_vsx());
    }

    return s.c_str();
}
