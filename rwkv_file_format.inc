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
