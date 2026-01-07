from tinygrad_scratch.engine import Value


def analytic_grads(f, inputs):
    values = [Value(x) for x in inputs]
    out = f(values)
    out.backward()
    return [v.grad for v in values]


def numeric_grads(f, inputs, eps=1e-6):
    grads = []
    for i in range(len(inputs)):
        up = [Value(x) for x in inputs]
        up[i] = Value(inputs[i] + eps)
        down = [Value(x) for x in inputs]
        down[i] = Value(inputs[i] - eps)
        grads.append((f(up).data - f(down).data) / (2 * eps))
    return grads


def gradcheck(f, inputs, tol=1e-5):
    a = analytic_grads(f, inputs)
    n = numeric_grads(f, inputs)
    ok = all(abs(x - y) <= tol for x, y in zip(a, n))
    return ok, a, n
