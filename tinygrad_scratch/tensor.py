import numpy as np


def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """An array that records operations, like Value but backed by numpy."""

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad = self.grad + _unbroadcast(out.grad, self.data.shape)
            other.grad = other.grad + _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = self.grad + _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad = other.grad + _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad = self.grad + out.grad @ other.data.T
            other.grad = other.grad + self.data.T @ out.grad

        out._backward = _backward
        return out

    __matmul__ = matmul

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward():
            self.grad = self.grad + (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad = self.grad + np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Tensor(shape={self.data.shape})"
