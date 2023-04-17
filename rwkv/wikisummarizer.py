# Provides terminal-based chat interface for RWKV model.

import os
import sys
import argparse
import pathlib
import sampling
import tokenizers
import cpp_model
import cpp_shared_library
import multiprocessing

def relpath(p):
    import os
    pathname = os.path.dirname(sys.argv[0])
    return os.path.join(pathname, p)
    
# ======================================== Script settings ========================================

# Copied from https://github.com/ggerganov/llama.cpp/blob/6e7801d08d81c931a5427bae46f00763e993f54a/prompts/chat-with-bob.txt
prompt: str = """
Transcript of an article summarization AI. The HTML content of the article is provided under section `Article`, and the AI responds with a summary of the article in plain text under section `Summary`.

### Article

""" + open(relpath("wikipedia-dataflow-programming.html")).read() + """
### Summary

"""

# No trailing space here!
bot_message_prefix: str = '### Sumary\n\n'
user_message_prefix: str = '### Article\n\n'

max_tokens_per_generation: int = 10000

# Sampling settings.
temperature: float = 0.8
top_p: float = 0.5

# =================================================================================================

parser = argparse.ArgumentParser(description='Provide terminal-based chat interface for RWKV model')
parser.add_argument('model_path', help='Path to RWKV model in ggml format')
args = parser.parse_args()

assert prompt != '', 'Prompt must not be empty'

print('Loading 20B tokenizer')
tokenizer_path = pathlib.Path(os.path.abspath(__file__)).parent / '20B_tokenizer.json'
tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

library = cpp_shared_library.load_rwkv_shared_library()
print(f'System info: {library.rwkv_get_system_info_string()}')

print('Loading RWKV model')
model = cpp_model.RWKVModel(library, args.model_path, multiprocessing.cpu_count())
prompt_tokens = tokenizer.encode(prompt).ids
prompt_token_count = len(prompt_tokens)
print(f'Processing {prompt_token_count} prompt tokens, may take a while')

logits, state = None, None

from tqdm import tqdm

for token in tqdm(prompt_tokens):
    logits, state = model.eval(token, state, state, logits)

print('\nChat initialized! Write something and press Enter.')

for i in range(max_tokens_per_generation):
    token = sampling.sample_logits(logits, temperature, top_p)

    decoded = tokenizer.decode([token])

    print(decoded, end='', flush=True)

    logits, state = model.eval(token, state, state, logits)

