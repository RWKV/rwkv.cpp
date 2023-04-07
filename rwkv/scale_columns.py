# Scales columns of matrices to 0..1 range for better quantization.
# After scaling, the model needs to be quantized with `quantize.py`.
# Usage: python scale_columns.py C:\rwkv.cpp-169M.bin C:\rwkv.cpp-169M-columns-scaled.bin

import struct
import argparse
import numpy as np
from typing import List

def parse_args():
    parser = argparse.ArgumentParser(description='Scale columns of matrices to 0..1 range for better quantization')
    parser.add_argument('src_path', help='Path to FP32/FP16 checkpoint file')
    parser.add_argument('dest_path', help='Path to resulting checkpoint file, will be overwritten')
    return parser.parse_args()

def write_tensor_header(out_file, key: str, shape: List[int], data_type: int) -> None:
    encoded_key: bytes = key.encode('utf-8')

    out_file.write(struct.pack(
        '=iii',
        len(shape),
        len(encoded_key),
        data_type
    ))

    for dim in reversed(shape):
        out_file.write(struct.pack('=i', dim))

    out_file.write(encoded_key)

def test_scale(max_by_column: np.ndarray, min_by_column: np.ndarray, normalized_parameter: np.ndarray, parameter: np.ndarray) -> None:
    """
    Tests that scaling did not break the matrix.
    """

    random_input: np.ndarray = np.random.random(np.shape(parameter)[1]) * 4 - 2.5

    in_by_max: np.ndarray = random_input * max_by_column
    min_dot_in: np.ndarray = np.dot(min_by_column, random_input)

    actual_result: np.ndarray = (normalized_parameter @ in_by_max) + min_dot_in
    expected_result: np.ndarray = parameter @ random_input

    diff_0: float = np.sum(actual_result - expected_result) / len(random_input)
    diff_1: float = np.sum(expected_result - actual_result) / len(random_input)

    assert diff_0 < 0.000001 or diff_1 < 0.000001, f'{diff_0}, {diff_1}'

def main() -> None:
    args = parse_args()

    with open(args.src_path, 'rb') as in_file:
        magic,\
            version,\
            n_vocab,\
            n_embed,\
            n_layer,\
            data_type = struct.unpack('=' + 'i' * 6, in_file.read(4 * 6))

        assert magic == 0x67676d66, 'Unexpected magic value'
        assert version == 100, 'Unexpected file version'
        assert data_type == 0 or data_type == 1, 'Only FP32 and FP16 models are supported'

        with open(args.dest_path, 'wb') as out_file:
            out_file.write(struct.pack(
                '=' + 'i' * 6,
                magic,
                version,
                n_vocab,
                n_embed,
                n_layer,
                data_type
            ))

            while True:
                dim_count_bytes: bytes = in_file.read(4)

                if len(dim_count_bytes) < 4:
                    # EOF
                    break

                dim_count: int = struct.unpack('=i', dim_count_bytes)[0]
                key_length: int = struct.unpack('=i', in_file.read(4))[0]
                data_type: int = struct.unpack('=i', in_file.read(4))[0]

                assert data_type == 0 or data_type == 1, 'Only FP32 and FP16 parameters are supported'

                shape: List[int] = [0] * dim_count

                element_count: int = 1

                for i in range(len(shape)):
                    shape[i] = struct.unpack('=i', in_file.read(4))[0]

                    element_count *= shape[i]

                # PyTorch/numpy order
                shape = [i for i in reversed(shape)]

                key: str = in_file.read(key_length).decode('utf-8')

                parameter = np.frombuffer(
                    in_file.read((4 if data_type == 0 else 2) * element_count),
                    dtype=(np.single if data_type == 0 else np.half)
                ).reshape(shape)

                # ---

                print(key)

                write_tensor_header(out_file, key, shape, data_type)

                if dim_count == 2 and key != 'emb.weight' and key != 'head.weight':
                    # Scale
                    min_by_column: np.ndarray = np.amin(parameter, axis=0)
                    normalized_parameter: np.ndarray = parameter - min_by_column
                    max_by_column: np.ndarray = np.amax(normalized_parameter, axis=0)
                    normalized_parameter: np.ndarray = normalized_parameter / max_by_column
                    normalized_parameter.tofile(out_file)

                    test_scale(max_by_column, min_by_column, normalized_parameter, parameter)

                    # Force these vectors to have FP32 type to make ggml happy
                    write_tensor_header(out_file, key + '.min_by_column', list(min_by_column.shape), 0)
                    np.float32(min_by_column).tofile(out_file)

                    write_tensor_header(out_file, key + '.max_by_column', list(max_by_column.shape), 0)
                    np.float32(max_by_column).tofile(out_file)
                else:
                    # Write values as-is
                    parameter.tofile(out_file)

    print('Done')

if __name__ == "__main__":
    main()
