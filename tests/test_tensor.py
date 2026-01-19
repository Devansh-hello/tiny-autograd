import numpy as np
import pytest

from tinygrad_scratch.tensor import Tensor

torch = pytest.importorskip("torch")


def check_grads(expr_ours, expr_torch, arrays):
    ours = [Tensor(a) for a in arrays]
    out = expr_ours(ours)
    out.backward()

    ts = [torch.tensor(a, dtype=torch.double, requires_grad=True) for a in arrays]
    tout = expr_torch(ts)
    tout.backward()

    for o, t in zip(ours, ts):
        assert np.allclose(o.grad, t.grad.numpy(), atol=1e-9)


def test_matmul_nonsquare():
    np.random.seed(0)
    a = np.random.randn(2, 3)
    b = np.random.randn(3, 4)
    check_grads(lambda v: (v[0] @ v[1]).sum(), lambda t: (t[0] @ t[1]).sum(), [a, b])


def test_broadcast_add_bias():
    np.random.seed(1)
    x = np.random.randn(4, 5)
    bias = np.random.randn(5)
    check_grads(
        lambda v: (v[0] + v[1]).relu().sum(),
        lambda t: (t[0] + t[1]).relu().sum(),
        [x, bias],
    )


def test_mul_broadcast_row():
    np.random.seed(2)
    x = np.random.randn(3, 2)
    row = np.random.randn(1, 2)
    check_grads(lambda v: (v[0] * v[1]).sum(), lambda t: (t[0] * t[1]).sum(), [x, row])
