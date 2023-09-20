# Generates some text with an RWKV model.
# Usage example: python inference_example.py C:\rwkv.cpp-169M-Q5_1.bin 20B

import argparse
import sampling
from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
from tokenizer_util import get_tokenizer

# Parse received arguments.
parser = argparse.ArgumentParser(description='Generate some text with an RWKV model')
parser.add_argument('model_path', help='Path to RWKV model in ggml format')
parser.add_argument('tokenizer', help='Tokenizer to use; supported tokenizers: 20B, world', nargs='?', type=str, default='20B')
args = parser.parse_args()

# Load the model.
library = rwkv_cpp_shared_library.load_rwkv_shared_library()
model = rwkv_cpp_model.RWKVModel(library, args.model_path)

# Set up the tokenizer.
tokenizer_decode, tokenizer_encode = get_tokenizer(args.tokenizer)

# Prepare the prompt.
prompt: str = """One upon a time,"""
prompt_tokens = tokenizer_encode(prompt)

# Process the prompt.
init_logits, init_state = None, None

for token in prompt_tokens:
    init_logits, init_state = model.eval(token, init_state, init_state, init_logits)

logits, state = init_logits.clone(), init_state.clone()

# Generate and print the completion.
print(prompt, end='')

for i in range(32):
    token = sampling.sample_logits(logits, temperature=0.8, top_p=0.5)

    print(tokenizer_decode([token]), end='', flush=True)

    logits, state = model.eval(token, state, state, logits)

# Don't forget to free the memory after you are done working with the model!
model.free()
