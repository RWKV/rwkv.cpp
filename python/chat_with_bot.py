# Provides terminal-based chat interface for RWKV model.
# Usage: python chat_with_bot.py C:\rwkv.cpp-169M.bin
# Prompts and code adapted from https://github.com/BlinkDL/ChatRWKV/blob/9ca4cdba90efaee25cfec21a0bae72cbd48d8acd/chat.py

import os
import argparse
import pathlib
import copy
import json
import time
import sampling
from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
from tokenizer_util import add_tokenizer_argument, get_tokenizer
from typing import List, Dict, Optional

# ======================================== Script settings ========================================

# English, Chinese, Japanese
LANGUAGE: str = 'English'
# QA: Question and Answer prompt to talk to an AI assistant.
# Chat: chat prompt (need a large model for adequate quality, 7B+).
PROMPT_TYPE: str = 'QA'

MAX_GENERATION_LENGTH: int = 250

# Sampling temperature. It could be a good idea to increase temperature when top_p is low.
TEMPERATURE: float = 0.8
# For better Q&A accuracy and less diversity, reduce top_p (to 0.5, 0.2, 0.1 etc.)
TOP_P: float = 0.5
# Penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
PRESENCE_PENALTY: float = 0.2
# Penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
FREQUENCY_PENALTY: float = 0.2

END_OF_LINE_TOKEN: int = 187
DOUBLE_END_OF_LINE_TOKEN: int = 535
END_OF_TEXT_TOKEN: int = 0

# =================================================================================================

parser = argparse.ArgumentParser(description='Provide terminal-based chat interface for RWKV model')
parser.add_argument('model_path', help='Path to RWKV model in ggml format')
add_tokenizer_argument(parser)
args = parser.parse_args()

script_dir: pathlib.Path = pathlib.Path(os.path.abspath(__file__)).parent

with open(script_dir / 'prompt' / f'{LANGUAGE}-{PROMPT_TYPE}.json', 'r', encoding='utf8') as json_file:
    prompt_data = json.load(json_file)

    user, assistant, separator, init_prompt = prompt_data['user'], prompt_data['assistant'], prompt_data['separator'], prompt_data['prompt']

if init_prompt == '':
    raise ValueError('Prompt must not be empty')

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
print(f'System info: {library.rwkv_get_system_info_string()}')

print('Loading RWKV model')
model = rwkv_cpp_model.RWKVModel(library, args.model_path)

tokenizer_decode, tokenizer_encode = get_tokenizer(args.tokenizer, model.n_vocab)

# =================================================================================================

processed_tokens: List[int] = []
logits: Optional[rwkv_cpp_model.NumpyArrayOrPyTorchTensor] = None
state: Optional[rwkv_cpp_model.NumpyArrayOrPyTorchTensor] = None

def process_tokens(_tokens: List[int], new_line_logit_bias: float = 0.0) -> None:
    global processed_tokens, logits, state

    logits, state = model.eval_sequence_in_chunks(_tokens, state, state, logits, use_numpy=True)

    processed_tokens += _tokens

    logits[END_OF_LINE_TOKEN] += new_line_logit_bias

state_by_thread: Dict[str, Dict] = {}

def save_thread_state(_thread: str) -> None:
    state_by_thread[_thread] = {
        'tokens': copy.deepcopy(processed_tokens),
        'logits': copy.deepcopy(logits),
        'state': copy.deepcopy(state)
    }

def load_thread_state(_thread: str) -> None:
    global processed_tokens, logits, state

    thread_state = state_by_thread[_thread]

    processed_tokens = copy.deepcopy(thread_state['tokens'])
    logits = copy.deepcopy(thread_state['logits'])
    state = copy.deepcopy(thread_state['state'])

# Model only saw '\n\n' as [187, 187] before, but the tokenizer outputs [535] for it at the end.
# See https://github.com/BlinkDL/ChatRWKV/pull/110/files
def split_last_end_of_line(tokens: List[int]) -> List[int]:
    if len(tokens) > 0 and tokens[-1] == DOUBLE_END_OF_LINE_TOKEN:
        tokens = tokens[:-1] + [END_OF_LINE_TOKEN, END_OF_LINE_TOKEN]

    return tokens

# =================================================================================================

processing_start: float = time.time()

prompt_tokens = tokenizer_encode(init_prompt)
prompt_token_count = len(prompt_tokens)
print(f'Processing {prompt_token_count} prompt tokens, may take a while')

process_tokens(split_last_end_of_line(prompt_tokens))

processing_duration: float = time.time() - processing_start

print(f'Processed in {int(processing_duration)} s, {int(processing_duration / prompt_token_count * 1000)} ms per token')

save_thread_state('chat_init')
save_thread_state('chat')

print(f'\nChat initialized! Your name is {user}. Write something and press Enter. Use \\n to add line breaks to your message.')

while True:
    # Read user input
    user_input: str = input(f'> {user}{separator} ')
    msg: str = user_input.replace('\\n', '\n').strip()

    temperature: float = TEMPERATURE
    top_p: float = TOP_P

    if '-temp=' in msg:
        temperature = float(msg.split('-temp=')[1].split(' ')[0])

        msg = msg.replace('-temp='+f'{temperature:g}', '')

        if temperature <= 0.2:
            temperature = 0.2

        if temperature >= 5:
            temperature = 5

    if '-top_p=' in msg:
        top_p = float(msg.split('-top_p=')[1].split(' ')[0])

        msg = msg.replace('-top_p='+f'{top_p:g}', '')

        if top_p <= 0:
            top_p = 0

    msg = msg.strip()

    # + reset --> reset chat
    if msg == '+reset':
        load_thread_state('chat_init')
        save_thread_state('chat')
        print(f'{assistant}{separator} Chat reset.\n')
        continue
    elif msg[:5].lower() == '+gen ' or msg[:3].lower() == '+i ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

        # +gen YOUR PROMPT --> free single-round generation with any prompt. Requires Novel model.
        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            state = None
            processed_tokens = []
            process_tokens(tokenizer_encode(new))
            save_thread_state('gen_0')

        # +i YOUR INSTRUCT --> free single-round generation with any instruct. Requires Raven model.
        elif msg[:3].lower() == '+i ':
            new = f'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{msg[3:].strip()}

# Response:
'''
            state = None
            processed_tokens = []
            process_tokens(tokenizer_encode(new))
            save_thread_state('gen_0')

        # +qq YOUR QUESTION --> answer an independent question with more creativity (regardless of context).
        elif msg[:4].lower() == '+qq ':
            new = '\nQ: ' + msg[4:].strip() + '\nA:'
            state = None
            processed_tokens = []
            process_tokens(tokenizer_encode(new))
            save_thread_state('gen_0')

        # +qa YOUR QUESTION --> answer an independent question (regardless of context).
        elif msg[:4].lower() == '+qa ':
            load_thread_state('chat_init')

            real_msg = msg[4:].strip()
            new = f'{user}{separator} {real_msg}\n\n{assistant}{separator}'

            process_tokens(tokenizer_encode(new))
            save_thread_state('gen_0')

        # +++ --> continue last free generation (only for +gen / +i)
        elif msg.lower() == '+++':
            try:
                load_thread_state('gen_1')
                save_thread_state('gen_0')
            except Exception as e:
                print(e)
                continue

        # ++ --> retry last free generation (only for +gen / +i)
        elif msg.lower() == '++':
            try:
                load_thread_state('gen_0')
            except Exception as e:
                print(e)
                continue
        thread = 'gen_1'

    else:
        # + --> alternate chat reply
        if msg.lower() == '+':
            try:
                load_thread_state('chat_pre')
            except Exception as e:
                print(e)
                continue
        # chat with assistant
        else:
            load_thread_state('chat')
            new = f'{user}{separator} {msg}\n\n{assistant}{separator}'
            process_tokens(tokenizer_encode(new), new_line_logit_bias=-999999999)
            save_thread_state('chat_pre')

        thread = 'chat'

        # Print assistant response
        print(f'> {assistant}{separator}', end='')

    start_index: int = len(processed_tokens)
    accumulated_tokens: List[int] = []
    token_counts: Dict[int, int] = {}

    for i in range(MAX_GENERATION_LENGTH):
        for n in token_counts:
            logits[n] -= PRESENCE_PENALTY + token_counts[n] * FREQUENCY_PENALTY

        token: int = sampling.sample_logits(logits, temperature, top_p)

        if token == END_OF_TEXT_TOKEN:
            print()
            break

        if token not in token_counts:
            token_counts[token] = 1
        else:
            token_counts[token] += 1

        process_tokens([token])

        # Avoid UTF-8 display issues
        accumulated_tokens += [token]

        decoded: str = tokenizer_decode(accumulated_tokens)

        if '\uFFFD' not in decoded:
            print(decoded, end='', flush=True)

            accumulated_tokens = []

        if thread == 'chat':
            if '\n\n' in tokenizer_decode(processed_tokens[start_index:]):
                break

        if i == MAX_GENERATION_LENGTH - 1:
            print()

    save_thread_state(thread)
