"""Week 3: simulate two tensor-parallel ranks on one device.

Run: python week-3-tp-exercise.py
Requires PyTorch. Uses CPU float64 for correctness, not GPU performance.
Reference: https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html
"""
import torch
from torch.nn.functional import gelu


def tp_mlp(X, W1, W2, b1, b2):
    """Column-shard W1 and row-shard W2 along the same intermediate width."""
    assert W1.shape[1] % 2 == 0, 'This exercise uses equal intermediate shards.'
    W1a, W1b = W1.chunk(2, dim=1)
    W2a, W2b = W2.chunk(2, dim=0)
    b1a, b1b = b1.chunk(2)
    Ua = gelu(X @ W1a + b1a) @ W2a  # logical rank 0
    Ub = gelu(X @ W1b + b1b) @ W2b  # logical rank 1
    return Ua + Ub + b2  # a distributed version uses AllReduce(SUM) first


def main():
    torch.manual_seed(7)
    for M, D, F in [(2, 4, 8), (7, 6, 10), (1, 8, 12)]:
        X, W1, W2, b1, b2 = [
            torch.randn(shape, dtype=torch.float64)
            for shape in [(M, D), (D, F), (F, D), (F,), (D,)]
        ]
        reference = gelu(X @ W1 + b1) @ W2 + b2
        actual = tp_mlp(X, W1, W2, b1, b2)
        torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
        print(f'PASS: M={M}, D={D}, F={F}; max error={(actual-reference).abs().max().item():.3g}')

    # The output bias is added once, after the partial products are summed.
    # Adding b2 to both local partials before summing would add it twice.
    a = torch.tensor([1.0], dtype=torch.float64)
    assert not torch.allclose(gelu(a + a), gelu(a) + gelu(a))
    print('GELU is nonlinear: reduction-dimension partials must combine before GELU.')


if __name__ == '__main__':
    main()
