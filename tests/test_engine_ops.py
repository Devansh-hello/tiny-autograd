import math

from tinygrad_scratch.engine import Value


def test_add_grad():
    a = Value(2.0)
    b = Value(5.0)
    out = a + b
    out.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_mul_grad():
    a = Value(2.0)
    b = Value(5.0)
    out = a * b
    out.backward()
    assert a.grad == 5.0
    assert b.grad == 2.0


def test_pow_grad():
    a = Value(3.0)
    out = a ** 3
    out.backward()
    assert a.grad == 3 * 3.0 ** 2


def test_div_grad():
    a = Value(6.0)
    b = Value(2.0)
    out = a / b
    out.backward()
    assert a.grad == 1 / 2.0
    assert b.grad == -6.0 / 2.0 ** 2


def test_relu_grad_positive():
    a = Value(4.0)
    out = a.relu()
    out.backward()
    assert out.data == 4.0
    assert a.grad == 1.0


def test_relu_grad_negative():
    a = Value(-4.0)
    out = a.relu()
    out.backward()
    assert out.data == 0.0
    assert a.grad == 0.0


def test_tanh_grad():
    a = Value(0.7)
    out = a.tanh()
    out.backward()
    assert a.grad == 1 - math.tanh(0.7) ** 2


def test_exp_grad():
    a = Value(1.5)
    out = a.exp()
    out.backward()
    assert a.grad == math.exp(1.5)


def test_log_grad():
    a = Value(2.0)
    out = a.log()
    out.backward()
    assert a.grad == 1 / 2.0
