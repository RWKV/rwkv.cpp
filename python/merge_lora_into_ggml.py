# Merges a LoRA checkpoint in PyTorch format (.pth) into an rwkv.cpp model file.
# Usage: python merge_lora_into_ggml.py C:\rwkv.cpp-169M.bin C:\my-lora.pth 32 C:\rwkv.cpp-169M-with-my-lora.bin
# LoRA format is compatible with https://github.com/Blealtan/RWKV-LM-LoRA
# You need to know lora_alpha value to perform the merge.
# Source model must be in either FP16 or FP32 format. Quantization can be performed after merging.

import argparse
import struct
import torch
import numpy as np
from typing import List, Dict, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description='Merge a PyTorch LoRA checkpoint (.pth) into an rwkv.cpp model file')
    parser.add_argument('src_path', help='Path to source rwkv.cpp model')
    parser.add_argument('rwkv_arch_version', help='Version of RWKV architecture: v4, v5.1, v5.2, v6.0', type=str, choices=['v4', 'v5.1', 'v5.2', 'v6.0'])
    parser.add_argument('lora_path', help='Path to LoRA checkpoint in PyTorch format')
    parser.add_argument('lora_alpha', help='Value of lora_alpha parameter used when training this LoRA checkpoint', type=int)
    parser.add_argument('dest_path', help='Path to destination rwkv.cpp model, will be overwitten with the merged model')
    return parser.parse_args()

def write_parameter(out_file, key: str, parameter: torch.Tensor) -> None:
    assert parameter.dtype == torch.float32 or parameter.dtype == torch.float16

    key_encoded: bytes = key.encode('utf-8')

    out_file.write(struct.pack(
        '=iii',
        len(parameter.shape),
        len(key_encoded),
        1 if parameter.dtype == torch.float16 else 0
    ))

    # Dimension order is reversed here:
    # * PyTorch shape is (x rows, y columns)
    # * ggml shape is (y elements in a row, x elements in a column)
    # Both shapes represent the same tensor.
    for dim in reversed(parameter.shape):
        out_file.write(struct.pack('=i', dim))

    out_file.write(key_encoded)

    parameter.numpy().tofile(out_file)

def main() -> None:
    args = parse_args()

    arch_version: str = args.rwkv_arch_version

    if not (arch_version == 'v4' or arch_version == 'v5.1' or arch_version == 'v5.2' or arch_version == 'v6.0'):
        raise ValueError(f'Invalid RWKV architecture version {arch_version}')

    print(f'Reading {args.lora_path}')

    lora_state_dict: Dict[str, torch.Tensor] = torch.load(args.lora_path, map_location='cpu')

    print(f'Merging')

    with open(args.src_path, 'rb') as in_file, open(args.dest_path, 'wb') as out_file:
        # noinspection PyTypeChecker
        header: Tuple[int, int, int, int, int, int] = struct.unpack('=iiiiii', in_file.read(6 * 4))

        if header[0] != 0x67676d66:
            raise ValueError(f'Invalid magic value {header[0]:x}')
        if not (100 <= header[1] <= 101):
            raise ValueError(f'Invalid version number {header[1]}')
        if not (header[5] == 0 or header[5] == 1):
            raise ValueError('Only FP32 and FP16 models are supported')

        out_file.write(struct.pack('=iiiiii', *header))

        while True:
            parameter_header_bytes: bytes = in_file.read(3 * 4)

            if len(parameter_header_bytes) == 0:
                break

            dim_count, key_length, data_type = struct.unpack('=iii', parameter_header_bytes)

            # noinspection PyTypeChecker
            shape: Tuple[int] = struct.unpack('=' + 'i' * dim_count, in_file.read(dim_count * 4))
            # ggml order to PyTorch
            shape: List[int] = [d for d in reversed(shape)]

            key: str = in_file.read(key_length).decode('utf-8')

            print(f'* {key} {shape}')

            if not (data_type == 0 or data_type == 1):
                raise ValueError('Only FP32 and FP16 models are supported')

            element_count: int = 1

            for dim in shape:
                element_count *= dim

            parameter_np: np.ndarray = np.frombuffer(
                in_file.read((2 if data_type == 1 else 4) * element_count),
                dtype=(np.half if data_type == 1 else np.single)
            )

            parameter: torch.Tensor = torch.tensor(parameter_np).view(shape)

            if key in lora_state_dict:
                replacement: torch.Tensor = lora_state_dict[key].float()

                # Same processing as in convert_pytorch_to_ggml.py
                if '.time_' in key:
                    replacement = replacement.squeeze()

                if arch_version == 'v6.0':
                    if '.time_faaaa' in k:
                        replacement = replacement.unsqueeze(-1)
                    if '.time_maa_w1' in k or '.time_decay_w' in k:
                        replacement = replacement.transpose(0, 1)
                    if '.time_maa_w2' in k:
                        n_head: int = replacement.shape[1]
                        replacement = replacement.transpose(1, 2)
                    if '.time_decay' in k and '_w' not in k:
                        replacement = replacement.reshape(n_head, -1, 1)
                elif arch_version == 'v5.1' or arch_version == 'v5.2':
                    if '.time_decay' in key:
                        if arch_version == 'v5.2':
                            replacement = torch.exp(-torch.exp(replacement)).unsqueeze(-1)
                        else:
                            replacement = torch.exp(-torch.exp(replacement)).reshape(-1, 1, 1)

                    if '.time_first' in key:
                        replacement = torch.exp(replacement).reshape(-1, 1, 1)

                    if '.time_faaaa' in key:
                        replacement = replacement.unsqueeze(-1)
                else:
                    if '.time_decay' in key:
                        replacement = -torch.exp(replacement)

                if parameter.dtype == torch.float16:
                    replacement = replacement.half()

                if replacement.shape != parameter.shape:
                    raise ValueError(f'Parameter {key} has shape {parameter.shape} in model file ' \
                                     f'and shape {replacement.shape} in LoRA file')

                parameter = replacement

                print(f'Replaced parameter {key}')

                del lora_state_dict[key]

            for suffix in ['.weight', '']:
                lora_A_key: str = key.replace('.weight', '') + '.lora_A' + suffix
                lora_B_key: str = key.replace('.weight', '') + '.lora_B' + suffix

                if lora_A_key in lora_state_dict:
                    lora_A: torch.Tensor = lora_state_dict[lora_A_key]
                    lora_B: torch.Tensor = lora_state_dict[lora_B_key]

                    if lora_B.shape[1] != lora_A.shape[0]:
                        raise ValueError(f'Invalid shape of LoRA matrices for {key}: ' \
                                         f'{lora_A.shape}, {lora_B.shape}')

                    lora_R: int = lora_B.shape[1]

                    replacement: torch.Tensor = parameter + lora_B @ lora_A * (args.lora_alpha / lora_R)

                    if parameter.dtype == torch.float16:
                        replacement = replacement.half()

                    parameter = replacement

                    print(f'Merged LoRA into parameter {key}, lora_r = {lora_R}')

                    del lora_state_dict[lora_A_key]
                    del lora_state_dict[lora_B_key]

                    break

            write_parameter(out_file, key, parameter)

        for key in lora_state_dict:
            print(f'WARNING: Unused parameter in LoRA state dict {key}')

    print('Done')

if __name__ == "__main__":
    main()
