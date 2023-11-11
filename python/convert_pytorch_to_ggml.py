# Converts an RWKV model checkpoint in PyTorch format to an rwkv.cpp compatible file.
# Usage: python convert_pytorch_to_ggml.py C:\RWKV-4-Pile-169M-20220807-8023.pth C:\rwkv.cpp-169M-FP16.bin FP16
# Get model checkpoints from https://huggingface.co/BlinkDL
# See FILE_FORMAT.md for the documentation on the file format.

import argparse
import struct
import torch
from typing import Dict

def parse_args():
    parser = argparse.ArgumentParser(description='Convert an RWKV model checkpoint in PyTorch format to an rwkv.cpp compatible file')
    parser.add_argument('src_path', help='Path to PyTorch checkpoint file')
    parser.add_argument('dest_path', help='Path to rwkv.cpp checkpoint file, will be overwritten')
    parser.add_argument('data_type', help='Data type, FP16 or FP32', type=str, choices=['FP16', 'FP32', 'float16', 'float32'], default='FP16')
    return parser.parse_args()

def get_layer_count(state_dict: Dict[str, torch.Tensor]) -> int:
    n_layer: int = 0

    while f'blocks.{n_layer}.ln1.weight' in state_dict:
        n_layer += 1

    assert n_layer > 0

    return n_layer

def write_state_dict(state_dict: Dict[str, torch.Tensor], dest_path: str, data_type: str) -> None:
    emb_weight: torch.Tensor = state_dict['emb.weight']

    n_layer: int = get_layer_count(state_dict)
    n_vocab: int = emb_weight.shape[0]
    n_embed: int = emb_weight.shape[1]

    is_v5_1_or_2: bool = 'blocks.0.att.ln_x.weight' in state_dict
    is_v5_2: bool = 'blocks.0.att.gate.weight' in state_dict

    if is_v5_2:
        print('Detected RWKV v5.2')
    elif is_v5_1_or_2:
        print('Detected RWKV v5.1')
    else:
        print('Detected RWKV v4')

    with open(dest_path, 'wb') as out_file:
        is_FP16: bool = data_type == 'FP16' or data_type == 'float16'

        out_file.write(struct.pack(
            # Disable padding with '='
            '=iiiiii',
            # Magic: 'ggmf' in hex
            0x67676d66,
            101,
            n_vocab,
            n_embed,
            n_layer,
            1 if is_FP16 else 0
        ))

        for k in state_dict.keys():
            tensor: torch.Tensor = state_dict[k].float()

            if '.time_' in k:
                tensor = tensor.squeeze()

            if is_v5_1_or_2:
                if '.time_decay' in k:
                    if is_v5_2:
                        tensor = torch.exp(-torch.exp(tensor)).unsqueeze(-1)
                    else:
                        tensor = torch.exp(-torch.exp(tensor)).reshape(-1, 1, 1)

                if '.time_first' in k:
                    tensor = torch.exp(tensor).reshape(-1, 1, 1)

                if '.time_faaaa' in k:
                    tensor = tensor.unsqueeze(-1)
            else:
                if '.time_decay' in k:
                    tensor = -torch.exp(tensor)

            # Keep 1-dim vectors and small matrices in FP32
            if is_FP16 and len(tensor.shape) > 1 and '.time_' not in k:
                tensor = tensor.half()

            shape = tensor.shape

            print(f'Writing {k}, shape {shape}, type {tensor.dtype}')

            k_encoded: bytes = k.encode('utf-8')

            out_file.write(struct.pack(
                '=iii',
                len(shape),
                len(k_encoded),
                1 if tensor.dtype == torch.float16 else 0
            ))

            # Dimension order is reversed here:
            # * PyTorch shape is (x rows, y columns)
            # * ggml shape is (y elements in a row, x elements in a column)
            # Both shapes represent the same tensor.
            for dim in reversed(tensor.shape):
                out_file.write(struct.pack('=i', dim))

            out_file.write(k_encoded)

            tensor.numpy().tofile(out_file)

def main() -> None:
    args = parse_args()

    print(f'Reading {args.src_path}')

    state_dict: Dict[str, torch.Tensor] = torch.load(args.src_path, map_location='cpu')

    write_state_dict(state_dict, args.dest_path, args.data_type)

    print('Done')

if __name__ == "__main__":
    main()
