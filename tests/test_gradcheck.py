from gradcheck import gradcheck


def test_gradcheck_arithmetic():
    f = lambda v: v[0] * v[1] + v[0] / v[1] - v[1] ** 3
    ok, analytic, numeric = gradcheck(f, [2.0, -3.0])
    assert ok, (analytic, numeric)


def test_gradcheck_activations():
    f = lambda v: (v[0].tanh() + v[1].relu()) * v[2].exp()
    ok, analytic, numeric = gradcheck(f, [0.5, 1.2, -0.3])
    assert ok, (analytic, numeric)


def test_gradcheck_log_and_pow():
    f = lambda v: (v[0] ** 2 + 1.0).log() + v[1] ** 4
    ok, analytic, numeric = gradcheck(f, [1.5, 0.8])
    assert ok, (analytic, numeric)


def test_gradcheck_mixed():
    f = lambda v: ((v[0] * v[1]).tanh() + v[0]).relu() * (v[1] + 2.0)
    ok, analytic, numeric = gradcheck(f, [0.4, 0.9])
    assert ok, (analytic, numeric)
