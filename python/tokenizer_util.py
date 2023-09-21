import os
import pathlib
from rwkv_cpp import rwkv_world_tokenizer
from typing import List, Tuple, Callable

def add_tokenizer_argument(parser) -> None:
    parser.add_argument(
            'tokenizer',
            help='Tokenizer to use; supported tokenizers: auto (guess from n_vocab), 20B, world',
            nargs='?',
            type=str,
            default='auto'
    )

def get_tokenizer(tokenizer_name: str, n_vocab: int) -> Tuple[
    Callable[[List[int]], str],
    Callable[[str], List[int]]
]:
    if tokenizer_name == 'auto':
        if n_vocab == 50277:
            tokenizer_name = '20B'
        elif n_vocab == 65536:
            tokenizer_name = 'world'
        else:
            assert False, f'Can not guess the tokenizer from n_vocab value of {n_vocab}'

    parent: pathlib.Path = pathlib.Path(os.path.abspath(__file__)).parent

    if tokenizer_name == 'world':
        print('Loading World v20230424 tokenizer')
        return rwkv_world_tokenizer.get_world_tokenizer_v20230424()
    elif tokenizer_name == '20B':
        print('Loading 20B tokenizer')
        import tokenizers
        tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_file(str(parent / '20B_tokenizer.json'))
        return tokenizer.decode, lambda x: tokenizer.encode(x).ids
    else:
        assert False, f'Unknown tokenizer {tokenizer_name}'
