# import os
# import pathlib
# import copy
# import torch
# from rwkv_cpp.rwkv import sampling
# import tokenizers
# from rwkv_cpp.rwkv import rwkv_cpp_model
# from rwkv_cpp.rwkv import rwkv_cpp_shared_library
# import json
# from typing import Optional

# LANGUAGE: str = 'English'
# PROMPT_TYPE: str = 'QA'
# MAX_GENERATION_LENGTH: int = 250
# TEMPERATURE: float = 0.8
# TOP_P: float = 0.5
# PRESENCE_PENALTY: float = 0.2
# FREQUENCY_PENALTY: float = 0.2
# END_OF_LINE_TOKEN: int = 187
# END_OF_TEXT_TOKEN: int = 0
# MAX_CONVERSATION: int = 200

# filename='rwkv_cpp/rwkv/prompt/English-QA.json'
# script_dir = 'rwkv_cpp/rwkv'
# with open(filename, 'r', encoding='utf8') as json_file:
#     prompt_data = json.load(json_file)

#     user, bot, separator, init_prompt = prompt_data['user'], prompt_data['bot'], prompt_data['separator'], prompt_data['prompt']

# print('Loading 20B tokenizer')
# tokenizer_path = script_dir + '/20B_tokenizer.json'
# tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

# library = rwkv_cpp_shared_library.load_rwkv_shared_library()
# print(f'System info: {library.rwkv_get_system_info_string()}')

# print('Loading RWKV model')
# model = rwkv_cpp_model.RWKVModel(library, '/home/ubuntu/repos/mvp/Q8_0-RWKV-4-Raven-14B-v11x-Eng99%-Other1%-20230501-ctx8192.bin')

# prompt_tokens = tokenizer.encode(init_prompt).ids
# prompt_token_count = len(prompt_tokens)
# print(prompt_tokens)

# processed_tokens: list[int] = []
# logits: Optional[torch.Tensor] = None
# state: Optional[torch.Tensor] = None

# def process_tokens(_tokens: list[int], new_line_logit_bias: float = 0.0) -> None:
#     global processed_tokens, logits, state

#     processed_tokens += _tokens

#     for _token in _tokens:
#         logits, state = model.eval(_token, state, state, logits)

#     logits[END_OF_LINE_TOKEN] += new_line_logit_bias

# state_by_thread: dict[str, dict] = {}

# def save_thread_state(_thread: str) -> None:
#     state_by_thread[_thread] = {
#         'tokens': copy.deepcopy(processed_tokens),
#         'logits': copy.deepcopy(logits),
#         'state': copy.deepcopy(state)
#     }

# def load_thread_state(_thread: str) -> None:
#     global processed_tokens, logits, state

#     thread_state = state_by_thread[_thread]

#     processed_tokens = copy.deepcopy(thread_state['tokens'])
#     logits = copy.deepcopy(thread_state['logits'])
#     state = copy.deepcopy(thread_state['state'])

# print(f'Processing {prompt_token_count} prompt tokens, may take a while')

# process_tokens(tokenizer.encode(init_prompt).ids)

# save_thread_state('chat_init')
# save_thread_state('chat')

# print(f'\nChat initialized! Your name is {user}. Write something and press Enter. Use \\n to add line breaks to your message.')


# def process_chat(self):

#     # Read user input
#     user_input = input(f'> {user}{separator} ')
#     msg = user_input.replace('\\n', '\n').strip()

#     temperature = TEMPERATURE
#     top_p = TOP_P

#     load_thread_state('chat')
#     new = f'{user}{separator} {msg}\n\n{bot}{separator}'
#     process_tokens(tokenizer.encode(new).ids, new_line_logit_bias=-999999999)
#     save_thread_state('chat_pre')

#     thread = 'chat'
#     print(f'> {bot}{separator}', end='')

#     start_index: int = len(processed_tokens)
#     accumulated_tokens: list[int] = []
#     token_counts: dict[int, int] = {}

#     for i in range(MAX_GENERATION_LENGTH):
#         for n in token_counts:
#             logits[n] -= PRESENCE_PENALTY + token_counts[n] * FREQUENCY_PENALTY

#         token: int = sampling.sample_logits(logits, temperature, top_p)

#         if token == END_OF_TEXT_TOKEN:
#             print()
#             break

#         if token not in token_counts:
#             token_counts[token] = 1
#         else:
#             token_counts[token] += 1

#         process_tokens([token])

#         # Avoid UTF-8 display issues
#         accumulated_tokens += [token]

#         decoded: str = tokenizer.decode(accumulated_tokens)

#         if '\uFFFD' not in decoded:
#             print(decoded, end='', flush=True)

#             accumulated_tokens = []

#         if thread == 'chat':
#             if '\n\n' in tokenizer.decode(processed_tokens[start_index:]):
#                 break

#         if i == MAX_GENERATION_LENGTH - 1:
#             print("111")

#     save_thread_state(thread)
import os
import pathlib
import copy
import torch
from rwkv_cpp.rwkv import sampling
import tokenizers
from rwkv_cpp.rwkv import rwkv_cpp_model
from rwkv_cpp.rwkv import rwkv_cpp_shared_library
import json
from typing import Optional

class AskRWKV:
    def __init__(self):        
        self.LANGUAGE: str = 'English'
        self.PROMPT_TYPE: str = 'QA'
        self.MAX_GENERATION_LENGTH: int = 250
        self.TEMPERATURE: float = 0.8
        self.TOP_P: float = 0.5
        self.PRESENCE_PENALTY: float = 0.2
        self.FREQUENCY_PENALTY: float = 0.2
        self.END_OF_LINE_TOKEN: int = 187
        self.END_OF_TEXT_TOKEN: int = 0
        self.MAX_CONVERSATION: int = 200
        self.filename='rwkv_cpp/rwkv/prompt/English-QA.json'
        self.script_dir = 'rwkv_cpp/rwkv'
        self.user, self.bot, self.separator, self.init_prompt = self.load_prompt()
        self.tokenizer, library, self.model, prompt_tokens, self.prompt_token_count = self.tokenize()
        self.processed_tokens: list[int] = []
        self.logits: Optional[torch.Tensor] = None
        self.state: Optional[torch.Tensor] = None
        self.state_by_thread: dict[str, dict] = {}
        print(f'Processing {self.prompt_token_count} prompt tokens, may take a while')
        self.process_tokens(self.tokenizer.encode(self.init_prompt).ids)
        self.save_thread_state('chat_init')
        self.save_thread_state('chat')
        print(f'\nChat initialized! Your name is {self.user}. Write something and press Enter. Use \\n to add line breaks to your message.')

    def load_prompt(self):
        with open(self.filename, 'r', encoding='utf8') as json_file:
            prompt_data = json.load(json_file)

            user, bot, separator, init_prompt = prompt_data['user'], prompt_data['bot'], prompt_data['separator'], prompt_data['prompt']
        return user, bot, separator, init_prompt 

    def tokenize(self):
        print('Loading 20B tokenizer')
        tokenizer_path = self.script_dir + '/20B_tokenizer.json'
        tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

        library = rwkv_cpp_shared_library.load_rwkv_shared_library()
        print(f'System info: {library.rwkv_get_system_info_string()}')

        print('Loading RWKV model')
        model = rwkv_cpp_model.RWKVModel(library, '/home/ubuntu/repos/mvp/Q8_0-RWKV-4-Raven-14B-v11x-Eng99%-Other1%-20230501-ctx8192.bin')

        prompt_tokens = tokenizer.encode(self.init_prompt).ids
        prompt_token_count = len(prompt_tokens)
        print(prompt_tokens)
        return tokenizer, library, model, prompt_tokens, prompt_token_count

    def process_tokens(self, _tokens: list[int], new_line_logit_bias: float = 0.0) -> None:
        self.processed_tokens += _tokens
        for _token in _tokens:
            self.logits, self.state = self.model.eval(_token, self.state, self.state, self.logits)
        self.logits[self.END_OF_LINE_TOKEN] += new_line_logit_bias

    def save_thread_state(self, _thread: str) -> None:
        self.state_by_thread[_thread] = {
            'tokens': copy.deepcopy(self.processed_tokens),
            'logits': copy.deepcopy(self.logits),
            'state': copy.deepcopy(self.state)
        }

    def load_thread_state(self, _thread: str) -> None:
        thread_state = self.state_by_thread[_thread]
        self.processed_tokens = copy.deepcopy(thread_state['tokens'])
        self.logits = copy.deepcopy(thread_state['logits'])
        self.state = copy.deepcopy(thread_state['state'])

    def process_chat(self, user_input):
        #user_input = input(f'> {self.user}{self.separator} ')
        msg = user_input.replace('\\n', '\n').strip()

        temperature = self.TEMPERATURE
        top_p = self.TOP_P

        self.load_thread_state('chat')
        new = f'{self.user}{self.separator} {msg}\n\n{self.bot}{self.separator}'
        self.process_tokens(self.tokenizer.encode(new).ids, new_line_logit_bias=-999999999)
        self.save_thread_state('chat_pre')

        self.thread = 'chat'
        print(f'> {self.bot}{self.separator}', end='')

        start_index: int = len(self.processed_tokens)
        accumulated_tokens: list[int] = []
        token_counts: dict[int, int] = {}
        output_string=""

        for i in range(self.MAX_GENERATION_LENGTH):
            for n in token_counts:
                self.logits[n] -= self.PRESENCE_PENALTY + token_counts[n] * self.FREQUENCY_PENALTY

            token: int = sampling.sample_logits(self.logits, temperature, top_p)

            if token == self.END_OF_TEXT_TOKEN:
                print()
                break

            if token not in token_counts:
                token_counts[token] = 1
            else:
                token_counts[token] += 1

            self.process_tokens([token])

            # Avoid UTF-8 display issues
            accumulated_tokens += [token]

            decoded: str = self.tokenizer.decode(accumulated_tokens)
            output_string += decoded
            print("decoded = {}".format(decoded))
            if '\uFFFD' not in decoded:
                print(decoded, end='', flush=True)

                accumulated_tokens = []

            if self.thread == 'chat':
                if '\n\n' in self.tokenizer.decode(self.processed_tokens[start_index:]):
                    break

            if i == self.MAX_GENERATION_LENGTH - 1:
                print("111")

        self.save_thread_state(self.thread)
        return output_string
ask_rwkv = AskRWKV()