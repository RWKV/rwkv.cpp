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
CHAT_LANG = 'Chinese' # English // Chinese

QA_PROMPT = True # True: Q & A prompt // False: chat prompt (need large model)

if CHAT_LANG == 'English':
    interface = ':'

    if QA_PROMPT:
        user = "User"
        bot = "Bot" # Or: 'The following is a verbose and detailed Q & A conversation of factual information.'
        init_prompt = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} french revolution what year

{bot}{interface} The French Revolution started in 1789, and lasted 10 years until 1799.

{user}{interface} 3+5=?

{bot}{interface} The answer is 8.

{user}{interface} guess i marry who ?

{bot}{interface} Only if you tell me more about yourself - what are your interests?

{user}{interface} solve for a: 9-a=2

{bot}{interface} The answer is a = 7, because 9 - 7 = 2.

{user}{interface} what is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''        
    else:
        user = "Bob"
        bot = "Alice"
        init_prompt = f'''
The following is a verbose detailed conversation between {user} and a young girl {bot}. {bot} is intelligent, friendly and cute. {bot} is likely to agree with {user}.

{user}{interface} Hello {bot}, how are you doing?

{bot}{interface} Hi {user}! Thanks, I'm fine. What about you?

{user}{interface} I am very good! It's nice to see you. Would you mind me chatting with you for a while?

{bot}{interface} Not at all! I'm listening.

'''

elif CHAT_LANG == 'Chinese':
    interface = ":"
    if QA_PROMPT:
        user = "Q"
        bot = "A"
        init_prompt = f'''
Expert Questions & Helpful Answers

Ask Research Experts

'''
    else:
        user = "Bob"
        bot = "Alice"
        init_prompt = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} what is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

{user}{interface} 企鹅会飞吗

{bot}{interface} 企鹅是不会飞的。它们的翅膀主要用于游泳和平衡，而不是飞行。

'''

FREE_GEN_LEN: int = 100

# Sampling settings.
GEN_TEMP: float = 0.8 # It could be a good idea to increase temp when top_p is low
GEN_TOP_P: float = 0.5 # Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)

# =================================================================================================

parser = argparse.ArgumentParser(description='Provide terminal-based chat interface for RWKV model')
parser.add_argument('model_path', help='Path to RWKV model in ggml format')
args = parser.parse_args()

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

def run_rnn(tokens: List[int]):
    global model_tokens, model_state, logits

    tokens = [int(x) for x in tokens]
    model_tokens += tokens

    for token in tokens:
        logits, model_state = model.eval(token, model_state, model_state, logits)
    
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

for token in prompt_tokens:
    logits, model_state = model.eval(token, model_state, model_state, logits)
    model_tokens.append(token)

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
            logits = run_rnn(tokenizer.encode(new).ids)
            save_all_stat('chat_pre', logits)
        
        thread = 'chat'

        # Print bot response
        print(f"> {bot}{interface}", end='')

    decoded = ''
    begin = len(model_tokens)
    out_last = begin

    for i in range(FREE_GEN_LEN):
        token = sampling.sample_logits(logits, temperature, top_p)
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
