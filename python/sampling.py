import numpy as np
from typing import Dict

# https://stackoverflow.com/a/50425683
def softmax(x: np.ndarray, axis: int):
    x -= x.max(axis=axis, keepdims=True)
    e: np.ndarray = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def sample_logits(out, temperature: float = 1.0, top_p: float = 0.8, logit_bias: Dict[int, float] = None) -> int:
    if hasattr(out, '__module__') and out.__module__ == 'torch':
        out = out.cpu().numpy()

    probs: np.ndarray = softmax(out, axis=-1)

    return sample_probs(probs, temperature, top_p, logit_bias)

def sample_probs(probs: np.ndarray, temperature: float = 1.0, top_p: float = 0.8, logit_bias: Dict[int, float] = None) -> int:
    assert 0.0 <= temperature, 'temperature'
    assert 0.0 <= top_p <= 1.0, 'top_p'

    if top_p == 0.0:
        top_p = 1.0

    if logit_bias is not None and len(logit_bias) > 0:
        logits: np.ndarray = np.log(probs)

        ids, values = zip(*logit_bias.items())
        logits[list(ids)] += values

        # Makes calculation more numerically stable, does not change the result
        logits -= logits.max(axis=-1, keepdims=True)

        probs = np.exp(logits) / np.sum(np.exp(logits))

    if temperature == 0.0:
        return np.argmax(probs).item()

    if top_p < 1.0:
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0

    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)

    probs = probs / np.sum(probs)

    return np.random.choice(a=len(probs), p=probs)
