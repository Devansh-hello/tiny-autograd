from tinygrad_scratch.engine import Value


def test_value_added_to_itself():
    a = Value(4.0)
    b = a + a
    b.backward()
    assert a.grad == 2.0


def test_value_used_twice_in_mul():
    a = Value(3.0)
    b = a * a
    b.backward()
    assert a.grad == 6.0


def test_diamond_graph():
    a = Value(-2.0)
    b = Value(3.0)
    d = a * b
    e = a + b
    f = d * e
    f.backward()
    assert a.grad == 2 * a.data * b.data + b.data ** 2
    assert b.grad == a.data ** 2 + 2 * a.data * b.data
