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
