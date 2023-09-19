import os
import tokenizers
import pathlib
from rwkv import rwkv_world_tokenizer
from typing import List, Tuple, Callable

def get_tokenizer(tokenizer: str = '20B') -> Tuple[
    Callable[[List[int]], str],
    Callable[[str], List[int]]
]:
    parent: pathlib.Path = pathlib.Path(os.path.abspath(__file__)).parent

    if tokenizer == 'world':
        print('Loading world tokenizer')
        return rwkv_world_tokenizer.get_world_tokenizer_v20230424()
    elif tokenizer == '20B':
        print('Loading 20B tokenizer')
        tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_file(str(parent / '20B_tokenizer.json'))
        return tokenizer.decode, lambda x: tokenizer.encode(x).ids
    else:
        assert False, f'Unknown tokenizer {tokenizer}'
