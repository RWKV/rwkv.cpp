# Provides terminal-based chat interface for RWKV model.

import os
import sys
import argparse
import pathlib
import copy
from typing import List
import sampling
import tokenizers
import rwkv_cpp_model
import rwkv_cpp_shared_library

# ======================================== Script settings ========================================

# Copied from https://github.com/BlinkDL/ChatRWKV/blob/9ca4cdba90efaee25cfec21a0bae72cbd48d8acd/chat.py#L92-L178
CHAT_LANG = 'English' # English // Chinese

QA_PROMPT = 2 # 1: [User & Bot] (Q&A) prompt // 2: [Bob & Alice] (chat) prompt

PROMPT_FILE = f'./rwkv/prompt/default/{CHAT_LANG}-{QA_PROMPT}.py'

def load_prompt(PROMPT_FILE):
    variables = {}
    with open(PROMPT_FILE, 'rb') as file:
        exec(compile(file.read(), PROMPT_FILE, 'exec'), variables)
    user, bot, interface, init_prompt = variables['user'], variables['bot'], variables['interface'], variables['init_prompt']
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
        init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n' + ('\n'.join(init_prompt)).strip() + '\n\n'
    return user, bot, interface, init_prompt

FREE_GEN_LEN: int = 100

# Sampling settings.
GEN_TEMP: float = 1.1 # It could be a good idea to increase temp when top_p is low
GEN_TOP_P: float = 0.7 # Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)
GEN_alpha_presence = 0.2 # Presence Penalty
GEN_alpha_frequency = 0.2 # Frequency Penalty
END_OF_LINE = 187
END_OF_TEXT = 0

# =================================================================================================

parser = argparse.ArgumentParser(description='Provide terminal-based chat interface for RWKV model')
parser.add_argument('model_path', help='Path to RWKV model in ggml format')
args = parser.parse_args()

user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)
assert init_prompt != '', 'Prompt must not be empty'

print('Loading 20B tokenizer')
tokenizer_path = pathlib.Path(os.path.abspath(__file__)).parent / '20B_tokenizer.json'
tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
print(f'System info: {library.rwkv_get_system_info_string()}')

print('Loading RWKV model')
model = rwkv_cpp_model.RWKVModel(library, args.model_path)

prompt_tokens = tokenizer.encode(init_prompt).ids
prompt_token_count = len(prompt_tokens)
print(f'Processing {prompt_token_count} prompt tokens, may take a while')


########################################################################################################

def run_rnn(tokens: List[int], newline_adj = 0):
    global model_tokens, model_state, logits

    model_tokens += tokens

    for token in tokens:
        logits, model_state = model.eval(token, model_state, model_state, logits)

    logits[END_OF_LINE] += newline_adj # adjust \n probability
    
    return logits

all_state = {}

def save_all_stat(thread: str, last_out):
    n = f'{thread}'
    all_state[n] = {}
    all_state[n]['logits'] = copy.deepcopy(last_out)
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)

def load_all_stat(thread: str):
    global model_tokens, model_state
    n = f'{thread}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return copy.deepcopy(all_state[n]['logits'])

########################################################################################################

model_tokens = []
logits, model_state = None, None

logits = run_rnn(tokenizer.encode(init_prompt).ids)

save_all_stat('chat_init', logits)
print('\nChat initialized! Write something and press Enter.')
save_all_stat('chat', logits)

while True:
    # Read user input
    user_input = input(f'> {user}{interface} ')
    msg = user_input.replace('\\n','\n').strip()

    temperature = GEN_TEMP
    top_p = GEN_TOP_P
    if ("-temp=" in msg):
        temperature = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+f'{temperature:g}', "")
        # print(f"temp: {temperature}")
    if ("-top_p=" in msg):
        top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+f'{top_p:g}', "")
        # print(f"top_p: {top_p}")
    if temperature <= 0.2:
        temperature = 0.2
    if temperature >= 5:
        temperature = 5
    if top_p <= 0:
        top_p = 0
    msg = msg.strip()

    # + reset --> reset chat
    if msg == '+reset':
        logits = load_all_stat('chat_init')
        save_all_stat('chat', logits)
        print(f'{bot}{interface} "Chat reset."\n')
        continue
    elif msg[:5].lower() == '+gen ' or msg[:3].lower() == '+i ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

        # +gen YOUR PROMPT --> free single-round generation with any prompt. Requires Novel model.
        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            logits = run_rnn(tokenizer.encode(new).ids)
            save_all_stat('gen_0', logits)

        # +i YOUR INSTRUCT --> free single-round generation with any instruct. Requires Raven model.
        elif msg[:3].lower() == '+i ':
            new = f'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{msg[3:].strip()}

# Response:
'''
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            logits = run_rnn(tokenizer.encode(new).ids)
            save_all_stat('gen_0', logits)

        # +qq YOUR QUESTION --> answer an independent question with more creativity (regardless of context).
        elif msg[:4].lower() == '+qq ':
            new = '\nQ: ' + msg[4:].strip() + '\nA:'
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            logits = run_rnn(tokenizer.encode(new).ids)
            save_all_stat('gen_0', logits)

        # +qa YOUR QUESTION --> answer an independent question (regardless of context).
        elif msg[:4].lower() == '+qa ':
            logits = load_all_stat('chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')
            
            logits = run_rnn(tokenizer.encode(new).ids)
            save_all_stat('gen_0', logits)

        # +++ --> continue last free generation (only for +gen / +i)
        elif msg.lower() == '+++':
            try:
                logits = load_all_stat('gen_1')
                save_all_stat('gen_0', logits)
            except Exception as e:
                print(e)
                continue

        # ++ --> retry last free generation (only for +gen / +i)
        elif msg.lower() == '++':
            try:
                logits = load_all_stat('gen_0')
            except Exception as e:
                print(e)
                continue
        thread = "gen_1"

    else:
        # + --> alternate chat reply
        if msg.lower() == '+':
            try:
                logits = load_all_stat('chat_pre')
            except Exception as e:
                print(e)
                continue
        # chat with bot
        else:
            logits = load_all_stat('chat')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            logits = run_rnn(tokenizer.encode(new).ids, newline_adj=-999999999)
            save_all_stat('chat_pre', logits)
        
        thread = 'chat'

        # Print bot response
        print(f"> {bot}{interface}", end='')

    decoded = ''
    begin = len(model_tokens)
    out_last = begin
    occurrence = {}
    for i in range(FREE_GEN_LEN):
        for n in occurrence:
            logits[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
        token = sampling.sample_logits(logits, temperature, top_p)
        if token == END_OF_TEXT:
            print()
            break
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        logits = run_rnn([token])
        decoded = tokenizer.decode(model_tokens[out_last:])
        if '\ufffd' not in decoded: # avoid utf-8 display issues
            print(decoded, end='', flush=True)
            out_last = begin + i + 1

        if thread == 'chat':
            send_msg = tokenizer.decode(model_tokens[begin:])
            if  '\n\n' in send_msg:
                send_msg = send_msg.strip()
                break
        if i == FREE_GEN_LEN - 1:
            print()

    save_all_stat(thread, logits)
