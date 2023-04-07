# Compares logits from rwkv.cpp implementation of RWKV with logits from reference implementation of RWKV.
# Reference logits were generated with RWKV-4-Pile-169M-20220807-8023.pth model in PyTorch.
# Reference implementation code: https://github.com/BlinkDL/ChatRWKV/blob/0d0abf181356c6f27501274cad18bdf28c83a45b/RWKV_in_150_lines.py
# Usage: python compare_with_reference_implementation.py C:\rwkv.cpp-169M.bin

import os
import struct
import argparse
import torch
import numpy as np
import rwkv_cpp_model
import rwkv_cpp_shared_library
from typing import List, Tuple, Any

def parse_args():
    parser = argparse.ArgumentParser(description='Compare logits from rwkv.cpp implementation of RWKV with logits from reference implementation of RWKV')
    parser.add_argument('ggml_model_path', help='Path to rwkv.cpp checkpoint file')
    return parser.parse_args()

def has_scaled_matrices(file_path: str) -> bool:
    with open(file_path, 'rb') as in_file:
        # Skip header
        in_file.read(4 * 6)

        while True:
            dim_count_bytes: bytes = in_file.read(4)

            if len(dim_count_bytes) < 4:
                # EOF
                break

            dim_count: int = struct.unpack('=i', dim_count_bytes)[0]
            key_length: int = struct.unpack('=i', in_file.read(4))[0]
            data_type: int = struct.unpack('=i', in_file.read(4))[0]

            shape: List[int] = [0] * dim_count

            element_count: int = 1

            for i in range(len(shape)):
                shape[i] = struct.unpack('=i', in_file.read(4))[0]

                element_count *= shape[i]

            key: str = in_file.read(key_length).decode('utf-8')

            byte_count: int

            if data_type == 0:
                byte_count = element_count * 4
            elif data_type == 1:
                byte_count = element_count * 2
            elif data_type == 2:
                byte_count = element_count // 32 * 20
            elif data_type == 3:
                byte_count = element_count // 32 * 24
            elif data_type == 4:
                byte_count = element_count // 32 * 24

            in_file.read(byte_count)

            if key.endswith('.min_by_column') or key.endswith('.max_by_column'):
                return True

            if key.startswith('blocks.1.'):
                # If we are already on second block, there are definetly
                # no scaled matrices, no need to read rest of the file
                return False

    return False

def main() -> None:
    args = parse_args()

    model_path: str = args.ggml_model_path

    print(f'Testing {model_path}')

    # Don't want to depend on tokenizer here.
    tokens: List[int] = [4, 3631, 4420, 2412, 953, 432, 391, 30567, 87, 15, 14161, 7092, 273, 416, 27767, 55, 342,
                         2412, 953, 432, 3806, 7092, 273, 416, 27767, 55, 15, 187, 4, 19039, 2412, 953, 497, 4561,
                         342, 416, 27767, 55, 14, 21, 14, 49, 587, 14, 17809, 46, 14, 938, 14256, 28950, 14, 1438,
                         1508, 15, 81, 394, 1566, 275, 8462, 22097, 348, 15, 187, 4, 43825, 27, 15548, 7277, 64,
                         3113, 64, 14005, 64, 39595, 15, 4789, 10269, 61, 18992, 61, 7265, 64, 30217, 39297, 15,
                         20963, 330, 27, 190, 30567, 87, 15, 14161, 14, 17809, 46, 15, 4805]

    threshold: float

    model_has_scaled_matrices: bool = has_scaled_matrices(model_path)

    with open(model_path, 'rb') as model_file:
        header: Tuple[Any] = struct.unpack('=iiiiii', model_file.read(6 * 4))
        data_type: int = header[5]

        assert data_type == 0 or\
               data_type == 1 or\
               data_type == 2 or\
               data_type == 3 or\
               data_type == 4, f'Unsupported model data type {data_type}'

        if data_type == 0:
            # FP32
            threshold = 0.00002 if model_has_scaled_matrices else 0.000005
        elif data_type == 1:
            # FP16
            threshold = 0.001 if model_has_scaled_matrices else 0.0032
        elif data_type == 2:
            # Q4_0
            threshold = 0.0054 if model_has_scaled_matrices else 0.4
        elif data_type == 3:
            # Q4_1
            threshold = 0.84 if model_has_scaled_matrices else 1.21
        elif data_type == 4:
            # Q4_1_O
            threshold = 0.32 if model_has_scaled_matrices else 0.2

    model = rwkv_cpp_model.RWKVModel(rwkv_cpp_shared_library.load_rwkv_shared_library(), args.ggml_model_path)

    def compare_logits(tokens_subset: List[int]) -> None:
        token_count: int = len(tokens_subset)

        logits, state = None, None

        for i in range(token_count):
            token: int = tokens_subset[i]

            if token_count <= 4 or i % (token_count // 4) == 0:
                print(f'{i + 1}/{token_count}')

            logits, state = model.eval(token, state, state, logits)

        actual_logits = logits

        # ---

        expected_logits_path: str = f'expected_logits_169M_20220807_8023_{len(tokens_subset)}_tokens.bin'

        if not os.path.isfile(expected_logits_path):
            expected_logits_path = 'rwkv/' + expected_logits_path

        with open(expected_logits_path, 'rb') as logits_file:
            expected_logits = torch.tensor(np.frombuffer(logits_file.read(), dtype=np.single))

        # ---

        difference: float = (torch.sum(expected_logits - actual_logits) / len(expected_logits)).item()

        print(f'Reference logits: {expected_logits}')
        print(f'Actual logits: {actual_logits}')
        print('Difference per token: %.8f' % (difference,))

        assert abs(difference) <= threshold, f'Difference {difference} is bigger than {threshold}'

    compare_logits(tokens)

    print()
    print('Test passes')

    if model is not None:
        model.free()

if __name__ == "__main__":
    main()
