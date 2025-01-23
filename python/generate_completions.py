# Generates completions from RWKV model based on a prompt.
# Usage example: python generate_completions.py C:\rwkv.cpp-169M-Q5_1.bin 20B

import argparse
import time
import sampling
from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
from tokenizer_util import add_tokenizer_argument, get_tokenizer
from typing import List

# ======================================== Script settings ========================================

prompt: str = """# rwkv.cpp

This is a port of [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) to [ggerganov/ggml](https://github.com/ggerganov/ggml).

Besides the usual **FP32**, it supports **FP16**, **quantized INT4, INT5 and INT8** inference. This project is **focused on CPU**, but cuBLAS is also supported."""

# How many completions to generate.
generation_count: int = 3
# Token count per single completion.
tokens_per_generation: int = 100

# Sampling settings.
temperature: float = 0.8
top_p: float = 0.5

# =================================================================================================

parser = argparse.ArgumentParser(description='Generate completions from RWKV model based on a prompt')
parser.add_argument('model_path', help='Path to RWKV model in ggml format')
parser.add_argument('-ngl', '--num_gpu_layers', type=int, default=99, help='Number of layers to run on GPU')
add_tokenizer_argument(parser)
args = parser.parse_args()

if prompt == '':
    raise ValueError('Prompt must not be empty')

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
print(f'System info: {library.rwkv_get_system_info_string()}')

print('Loading RWKV model')
model = rwkv_cpp_model.RWKVModel(library, args.model_path, gpu_layers_count=args.num_gpu_layers)

tokenizer_decode, tokenizer_encode = get_tokenizer(args.tokenizer, model.n_vocab)

prompt_tokens: List[int] = tokenizer_encode(prompt)

prompt_token_count: int = len(prompt_tokens)
print(f'{prompt_token_count} tokens in prompt')

init_logits, init_state = model.eval_sequence_in_chunks(prompt_tokens, None, None, None, use_numpy=True)

for GENERATION in range(generation_count):
    print(f'\n--- Generation {GENERATION} ---\n')
    print(prompt, end='[')

    start: float = time.time()

    logits, state = init_logits.copy(), init_state.copy()

    for i in range(tokens_per_generation):
        token: int = sampling.sample_logits(logits, temperature, top_p)

        print(tokenizer_decode([token]), end='', flush=True)

        logits, state = model.eval(token, state, state, logits, use_numpy=True)

    delay: float = time.time() - start

    print(']\n\nTook %.3f sec, %d ms per token' % (delay, delay / tokens_per_generation * 1000))
