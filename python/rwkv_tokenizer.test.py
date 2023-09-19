from rwkv_tokenizer import get_tokenizer
from typing import List

def test():
    tokenizer_decode, tokenizer_encode = get_tokenizer('world')

    test_string: str = 'I\'ll \'d test блабла 以下は、]) -> <|endoftext|><|padding|> int'

    expected_tokens: List[int] = [74, 5229, 274, 101, 32223, 5092, 27980, 2795, 27980, 33, 10399, 10258, 10139,
                                  10079, 1682, 3463, 295, 125, 25258, 7588, 2318, 125, 790, 125, 49520, 125, 63,
                                  21888]

    actual_tokens: List[int] = tokenizer_encode(test_string)
    assert actual_tokens == expected_tokens, f'\nActual: {actual_tokens}\nExpected: {expected_tokens}'

    decoded_string: str = tokenizer_decode(actual_tokens)
    assert test_string == decoded_string, f'\nDecoding mismatch: \n{decoded_string}'

    print('All tests pass')

if __name__ == "__main__":
    test()
