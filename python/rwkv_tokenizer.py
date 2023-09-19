import os
import tokenizers
import pathlib
from typing import List, Set, Tuple, Callable

# Taken from https://github.com/BlinkDL/ChatRWKV/tree/main/tokenizer/rwkv_tokenizer.py

class Trie:
    __slots__ = tuple('ch,to,values,front'.split(','))

    to: List
    values: Set

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for _ in range(256)]
        self.values = set()
        self.front = front

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key

            self.values.add(val)

            return self

        ch = key[idx]

        if self.to[ch] is None:
            self.to[ch] = Trie(front=self, ch=ch)

        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        u: Trie = self
        ch: int = key[idx]
        ret = None

        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1

            if u.values:
                ret = idx, u, u.values

            if idx == len(key):
                break

            ch = key[idx]

        assert ret is not None, 'Entry not found'

        return ret

    def __repr__(self):
        fr = self
        ret = []

        while fr != None:
            if fr.ch != None:
                ret.append(fr.ch)

            fr = fr.front

        return '<TRIE %s %s>' % (ret[::-1], self.values)

class WorldTokenizer:

    def __init__(self, file_path):
        self.index_to_token = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            idx = int(line[:line.index(' ')])
            x = eval(line[line.index(' '):line.rindex(' ')])
            x = x.encode('utf-8') if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(line[line.rindex(' '):])
            self.index_to_token[idx] = x

        self.token2idx = {}

        for k, v in self.index_to_token.items():
            self.token2idx[v] = int(k)

        self.root = Trie()

        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encode_bytes(self, src: bytes) -> List[int]:
        idx: int = 0
        tokens: List[int] = []

        while idx < len(src):
            _idx: int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert (idx != _idx)
            _, token = next(iter(values))
            tokens.append(token)

        return tokens

    def decode_bytes(self, tokens: List[int]) -> bytes:
        return b''.join(map(lambda i: self.index_to_token[i], tokens))

    def encode(self, src: str) -> List[int]:
        return self.encode_bytes(src.encode('utf-8'))

    def decode(self, tokens: List[int]) -> str:
        # 'replace' error handling mode will insert \uFFFD characters in place of malformed/partial UTF-8 sequences.
        # Downstream code needs to detect \uFFFD and attempt to postpone decoding until more tokens arrive and UTF-8 sequences are complete.
        return self.decode_bytes(tokens).decode('utf-8', errors='replace')

def get_tokenizer(tokenizer: str = '20B') -> Tuple[
    Callable[[List[int]], str],
    Callable[[str], List[int]]
]:
    parent: pathlib.Path = pathlib.Path(os.path.abspath(__file__)).parent

    if tokenizer == 'world':
        print('Loading world tokenizer')
        tokenizer: WorldTokenizer = WorldTokenizer(parent / 'rwkv_vocab_v20230424.txt')
        return tokenizer.decode, tokenizer.encode
    elif tokenizer == '20B':
        print('Loading 20B tokenizer')
        tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_file(str(parent / '20B_tokenizer.json'))
        return tokenizer.decode, lambda x: tokenizer.encode(x).ids
    else:
        assert False, f'Unknown tokenizer {tokenizer}'
