# rwkv.cpp file format

This format is used by `rwkv.cpp` to store RWKV model checkpoints.

Preferred file extension: `.bin`

Specification in C-like pseudocode:

```
RWKVModelFile {
    // All ints and floats are in machine byte order.
    // Magic is "ggml" string bytes.
    int32 magic = 0x67676d66;
    int32 version = 100;
    int32 n_vocab;
    int32 n_embed;
    int32 n_layer;
    // Data type of most of the parameters. See "Data types" below for possible values.
    int32 data_type;
    // Read until EOF.
    Parameter[] parameters;
}

Parameter {
    int32 dim_count;
    int32 key_length;
    // Data type of the parameter. See "Data types" below for possible values.
    int32 data_type;
    // Compared to PyTorch's parameter.shape, dimension order is reversed here!
    int32[dim_count] shape;
    // Keys are like "emb.weight", "block.0.ln1.weight".
    uint8[key_length] key_utf8;
    // Length of the data array depends on parameter data type:
    // - FP32: 4 * element_count 
    // - FP16: 2 * element_count
    // - QX_Y (quantized): element_count / QKX_Y * sizeof(block_qx_y)
    // See ggml.c for values of QK and block sizes of specific formats.
    byte[] data;
}
```

## Data types
 
- 0: `FP32`
- 1: `FP16`
- 2: `Q4_0`
- 3: `Q4_1`
- 4: *unused*
- 5: *unused*
- 6: *unused*
- 7: `Q5_0`
- 8: `Q5_1`
- 9: `Q8_0`
