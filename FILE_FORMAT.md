# rwkv.cpp file format

This format is used by `rwkv.cpp` to store RWKV model checkpoints.

Preferred file extension: `.bin`

Specification in C-like pseudocode:

```
RWKVModelFile {
    // All ints and floats are in machine byte order.
    // Magic is "ggml" string bytes.
    int32 magic = 0x67676d66;
    // Can be either 100 or 101. See "File versions" section below for details.
    int32 version = 101;
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

## File versions

### `100`

Original version number, chosen to not interfere with `llama.cpp` file version number of `1`.

### `101`

Introduced on 2023-05-27, as `ggml` was updated to commit [00b49ec](https://github.com/ggerganov/ggml/commit/00b49ec707d73df0176e21630a6e23c2aa0e938c).

All quantized formats (`QX_Y`) were changed in a backwards-incompatible way: new version of `ggml` can not handle loading version `100` quantized models.

`FP32` and `FP16` remain the same.

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
