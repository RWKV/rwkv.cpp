# Provides terminal-based chat interface for RWKV model.
# Usage: python chat_with_bot.py C:\rwkv.cpp-169M.bin
# Prompts and code adapted from https://github.com/BlinkDL/ChatRWKV/blob/9ca4cdba90efaee25cfec21a0bae72cbd48d8acd/chat.py

import os
import argparse
import pathlib
import copy
import torch
import sampling
import tokenizers
import rwkv_cpp_model
import rwkv_cpp_shared_library

# ======================================== Script settings ========================================

# English, Chinese
LANGUAGE: str = 'English'

# True: Q&A prompt
# False: chat prompt (you need a large model for adequate quality, 7B+)
QA_PROMPT: bool = False

MAX_GENERATION_LENGTH: int = 250

# Sampling temperature. It could be a good idea to increase temperature when top_p is low.
TEMPERATURE: float = 0.8
# For better Q&A accuracy and less diversity, reduce top_p (to 0.5, 0.2, 0.1 etc.)
TOP_P: float = 0.5

if LANGUAGE == 'English':
    separator: str = ':'

    if QA_PROMPT:
        user: str = 'User'
        bot: str = 'Bot'
        init_prompt: str = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and \
polite.

{user}{separator} french revolution what year

{bot}{separator} The French Revolution started in 1789, and lasted 10 years until 1799.

{user}{separator} 3+5=?

{bot}{separator} The answer is 8.

{user}{separator} guess i marry who ?

{bot}{separator} Only if you tell me more about yourself - what are your interests?

{user}{separator} solve for a: 9-a=2

{bot}{separator} The answer is a = 7, because 9 - 7 = 2.

{user}{separator} what is lhc

{bot}{separator} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''
    else:
        user: str = 'Bob'
        bot: str = 'Alice'
        init_prompt: str = f'''
The following is a verbose detailed conversation between {user} and a young girl {bot}. {bot} is intelligent, friendly and cute. {bot} is likely to agree with {user}.

{user}{separator} Hello {bot}, how are you doing?

{bot}{separator} Hi {user}! Thanks, I'm fine. What about you?

{user}{separator} I am very good! It's nice to see you. Would you mind me chatting with you for a while?

{bot}{separator} Not at all! I'm listening.

'''

elif LANGUAGE == 'Chinese':
    separator: str = ':'

    if QA_PROMPT:
        user: str = 'Q'
        bot: str = 'A'
        init_prompt: str = f'''
Expert Questions & Helpful Answers

Ask Research Experts

'''
    else:
        user: str = 'Bob'
        bot: str = 'Alice'
        init_prompt: str = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and \
polite.

{user}{separator} what is lhc

{bot}{separator} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

{user}{separator} 企鹅会飞吗

{bot}{separator} 企鹅是不会飞的。它们的翅膀主要用于游泳和平衡，而不是飞行。

'''
else:
    assert False, f'Invalid language {LANGUAGE}'

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

########################################################################################################

model_tokens: list[int] = []

logits, model_state = None, None

def process_tokens(_tokens: list[int]) -> torch.Tensor:
    global model_tokens, model_state, logits

    _tokens = [int(x) for x in _tokens]

    model_tokens += _tokens

    for _token in _tokens:
        logits, model_state = model.eval(_token, model_state, model_state, logits)

    return logits

state_by_thread: dict[str, dict] = {}

def save_thread_state(_thread: str, _logits: torch.Tensor) -> None:
    state_by_thread[_thread] = {}
    state_by_thread[_thread]['logits'] = copy.deepcopy(_logits)
    state_by_thread[_thread]['rnn'] = copy.deepcopy(model_state)
    state_by_thread[_thread]['token'] = copy.deepcopy(model_tokens)

def load_thread_state(_thread: str) -> torch.Tensor:
    global model_tokens, model_state
    model_state = copy.deepcopy(state_by_thread[_thread]['rnn'])
    model_tokens = copy.deepcopy(state_by_thread[_thread]['token'])
    return copy.deepcopy(state_by_thread[_thread]['logits'])

########################################################################################################

print(f'Processing {prompt_token_count} prompt tokens, may take a while')

for token in prompt_tokens:
    logits, model_state = model.eval(token, model_state, model_state, logits)

    model_tokens.append(token)

save_thread_state('chat_init', logits)
save_thread_state('chat', logits)

print(f'\nChat initialized! Your name is {user}. Write something and press Enter. Use \\n to add line breaks to your message.')

while True:
    # Read user input
    user_input = input(f'> {user}{separator} ')
    msg = user_input.replace('\\n', '\n').strip()

    temperature = TEMPERATURE
    top_p = TOP_P

    if "-temp=" in msg:
        temperature = float(msg.split('-temp=')[1].split(' ')[0])

        msg = msg.replace('-temp='+f'{temperature:g}', '')

        if temperature <= 0.2:
            temperature = 0.2

        if temperature >= 5:
            temperature = 5

    if "-top_p=" in msg:
        top_p = float(msg.split('-top_p=')[1].split(' ')[0])

        msg = msg.replace('-top_p='+f'{top_p:g}', '')

        if top_p <= 0:
            top_p = 0

    msg = msg.strip()

    # + reset --> reset chat
    if msg == '+reset':
        logits = load_thread_state('chat_init')
        save_thread_state('chat', logits)
        print(f'{bot}{separator} Chat reset.\n')
        continue
    elif msg[:5].lower() == '+gen ' or msg[:3].lower() == '+i ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

        # +gen YOUR PROMPT --> free single-round generation with any prompt. Requires Novel model.
        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            logits = process_tokens(tokenizer.encode(new).ids)
            save_thread_state('gen_0', logits)

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
            logits = process_tokens(tokenizer.encode(new).ids)
            save_thread_state('gen_0', logits)

        # +qq YOUR QUESTION --> answer an independent question with more creativity (regardless of context).
        elif msg[:4].lower() == '+qq ':
            new = '\nQ: ' + msg[4:].strip() + '\nA:'
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            logits = process_tokens(tokenizer.encode(new).ids)
            save_thread_state('gen_0', logits)

        # +qa YOUR QUESTION --> answer an independent question (regardless of context).
        elif msg[:4].lower() == '+qa ':
            logits = load_thread_state('chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{separator} {real_msg}\n\n{bot}{separator}"
            # print(f'### qa ###\n[{new}]')

            logits = process_tokens(tokenizer.encode(new).ids)
            save_thread_state('gen_0', logits)

        # +++ --> continue last free generation (only for +gen / +i)
        elif msg.lower() == '+++':
            try:
                logits = load_thread_state('gen_1')
                save_thread_state('gen_0', logits)
            except Exception as e:
                print(e)
                continue

        # ++ --> retry last free generation (only for +gen / +i)
        elif msg.lower() == '++':
            try:
                logits = load_thread_state('gen_0')
            except Exception as e:
                print(e)
                continue
        thread = "gen_1"

    else:
        # + --> alternate chat reply
        if msg.lower() == '+':
            try:
                logits = load_thread_state('chat_pre')
            except Exception as e:
                print(e)
                continue
        # chat with bot
        else:
            logits = load_thread_state('chat')
            new = f"{user}{separator} {msg}\n\n{bot}{separator}"
            # print(f'### add ###\n[{new}]')
            logits = process_tokens(tokenizer.encode(new).ids)
            save_thread_state('chat_pre', logits)

        thread = 'chat'

        # Print bot response
        print(f"> {bot}{separator}", end='')

    start_index: int = len(model_tokens)
    accumulated_tokens: list[int] = []

    for i in range(MAX_GENERATION_LENGTH):
        token: int = sampling.sample_logits(logits, temperature, top_p)

        logits: torch.Tensor = process_tokens([token])

        # Avoid UTF-8 display issues
        accumulated_tokens += [token]

        decoded: str = tokenizer.decode(accumulated_tokens)

        if '\uFFFD' not in decoded:
            print(decoded, end='', flush=True)

            accumulated_tokens = []

        if thread == 'chat':
            if '\n\n' in tokenizer.decode(model_tokens[start_index:]):
                break

        if i == MAX_GENERATION_LENGTH - 1:
            print()

    save_thread_state(thread, logits)
