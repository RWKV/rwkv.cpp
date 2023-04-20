# Provides terminal-based chat interface for RWKV model.

import os
import sys
import argparse
import pathlib
import copy
import sampling
import tokenizers
import rwkv_cpp_model
import rwkv_cpp_shared_library

# ======================================== Script settings ========================================

# Copied from https://github.com/BlinkDL/ChatRWKV/blob/9ca4cdba90efaee25cfec21a0bae72cbd48d8acd/chat.py#L92-L178
CHAT_LANG = 'English' # English // Chinese // more to come

QA_PROMPT = False # True: Q & A prompt // False: User & Bot prompt
# 中文问答设置QA_PROMPT=True（只能问答，问答效果更好，但不能闲聊） 中文聊天设置QA_PROMPT=False（可以闲聊，但需要大模型才适合闲聊）

if CHAT_LANG == 'English':
    interface = ":"

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

{user}{interface} wat is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''        
    else:
        user = "Bob"
        bot = "Alice"
        init_prompt = f'''
The following is a verbose detailed conversation between {user} and a young girl {bot}. {bot} is intelligent, friendly and cute. {bot} is unlikely to disagree with {user}.

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
        user = "User"
        bot = "Bot"
        init_prompt = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} what is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

{user}{interface} 企鹅会飞吗

{bot}{interface} 企鹅是不会飞的。它们的翅膀主要用于游泳和平衡，而不是飞行。

'''

max_tokens_per_generation: int = 100

# Sampling settings.
temperature: float = 0.8
top_p: float = 0.5

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
def run_rnn(tokens):
    global model_tokens, model_state, logits

    tokens = [int(x) for x in tokens]
    model_tokens += tokens

    for token in tokens:
        logits, model_state = model.eval(token, model_state, model_state, logits)
    
    return logits


all_state = {}
def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['logits'] = copy.deepcopy(last_out)
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)

def load_all_stat(srv, name):
    global model_tokens, model_state
    n = f'{name}_{srv}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return copy.deepcopy(all_state[n]['logits'])

########################################################################################################

model_tokens = []
logits, model_state = None, None

for token in prompt_tokens:
    logits, model_state = model.eval(token, model_state, model_state, logits)
    model_tokens.append(token)

save_all_stat('', 'chat_init', logits)
print('\nChat initialized! Write something and press Enter.')

srv_list = ['dummy_server']
for s in srv_list:
    save_all_stat(s, 'chat', logits)


def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')


while True:
    # Read user input
    user_input = input(f'> {user}{interface} ')
    msg = user_input.replace('\\n','\n').strip()

    srv = 'dummy_server'

    if msg == '+reset':
        logits = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', logits)
        reply_msg("Chat reset.")
        continue
    elif msg[:5].lower() == '+gen ' or msg[:3].lower() == '+i ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            logits = run_rnn(tokenizer.encode(new).ids)
            save_all_stat(srv, 'gen_0', logits)

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
            save_all_stat(srv, 'gen_0', logits)

        elif msg[:4].lower() == '+qq ':
            new = '\nQ: ' + msg[4:].strip() + '\nA:'
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            logits = run_rnn(tokenizer.encode(new).ids)
            save_all_stat(srv, 'gen_0', logits)

        elif msg[:4].lower() == '+qa ':
            logits = load_all_stat('', 'chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')
            
            logits = run_rnn(tokenizer.encode(new).ids)
            save_all_stat(srv, 'gen_0', logits)

        elif msg.lower() == '+++':
            try:
                logits = load_all_stat(srv, 'gen_1')
                save_all_stat(srv, 'gen_0', logits)
            except Exception as e:
                print(e)
                continue

        elif msg.lower() == '++':
            try:
                logits = load_all_stat(srv, 'gen_0')
            except Exception as e:
                print(e)
                continue
        name = "gen_1"
        print()

    else:
        if msg.lower() == '+':
            try:
                logits = load_all_stat(srv, 'chat_pre')
            except Exception as e:
                print(e)
                continue
        else:
            logits = load_all_stat(srv, 'chat')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            logits = run_rnn(tokenizer.encode(new).ids)
            save_all_stat(srv, 'chat_pre', logits)
        
        name = 'chat'

        # Generate and print bot response
        print(f"> {bot}{interface}", end='')

    decoded = ''
    begin = len(model_tokens)
    out_last = begin

    for i in range(max_tokens_per_generation):
        token = sampling.sample_logits(logits, temperature, top_p)
        logits = run_rnn([token])
        decoded = tokenizer.decode(model_tokens[out_last:])
        if '\ufffd' not in decoded: # avoid utf-8 display issues
            print(decoded, end='', flush=True)
            out_last = begin + i + 1

        if name == 'chat':
            send_msg = tokenizer.decode(model_tokens[begin:])
            if  '\n\n' in send_msg:
                send_msg = send_msg.strip()
                break

    print()
    save_all_stat(srv, name, logits)
