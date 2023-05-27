import os
import pathlib
import copy
import torch
from rwkv_cpp.rwkv import sampling
import tokenizers
from rwkv_cpp.rwkv import rwkv_cpp_model
from rwkv_cpp.rwkv import rwkv_cpp_shared_library
from rwkv_cpp.rwkv.rwkv_cpp_shared_library import RWKVSharedLibrary
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

    def load_prompt(self):
        with open(self.filename, 'r', encoding='utf8') as json_file:
            prompt_data = json.load(json_file)

            user, bot, separator, init_prompt = prompt_data['user'], prompt_data['bot'], prompt_data['separator'], prompt_data['prompt']
        return user, bot, separator, init_prompt 

    def tokenize(self):
        print('Loading 20B tokenizer')
        tokenizer_path = self.script_dir + '/20B_tokenizer.json'
        tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

        library = RWKVSharedLibrary(f'rwkv_cpp/rwkv/librwkv.so')
        print(f'System info: {library.rwkv_get_system_info_string()}')

        print('Loading RWKV model')
        model = rwkv_cpp_model.RWKVModel(library, 'raven.bin')

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
        msg = user_input.replace('\\n', '\n').strip()

        temperature = self.TEMPERATURE
        top_p = self.TOP_P

        self.load_thread_state('chat')
        new = f'{self.user}{self.separator} {msg}\n\n{self.bot}{self.separator}'
        self.process_tokens(self.tokenizer.encode(new).ids, new_line_logit_bias=-999999999)
        self.save_thread_state('chat_pre')

        self.thread = 'chat'

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
            if '\uFFFD' not in decoded:

                accumulated_tokens = []

            if self.thread == 'chat':
                if '\n\n' in self.tokenizer.decode(self.processed_tokens[start_index:]):
                    break

        self.save_thread_state(self.thread)
        return output_string
ask_rwkv = AskRWKV()