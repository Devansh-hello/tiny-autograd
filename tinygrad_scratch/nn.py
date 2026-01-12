import random

from tinygrad_scratch.engine import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, n_inputs, nonlin="tanh"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        if self.nonlin == "tanh":
            return act.tanh()
        if self.nonlin == "relu":
            return act.relu()
        return act

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_inputs, n_outputs, nonlin="tanh"):
        self.neurons = [Neuron(n_inputs, nonlin) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for n in self.neurons:
            params += n.parameters()
        return params


class MLP(Module):
    def __init__(self, n_inputs, layer_sizes, nonlin="tanh"):
        sizes = [n_inputs] + layer_sizes
        self.layers = []
        for i in range(len(layer_sizes)):
            is_last = i == len(layer_sizes) - 1
            self.layers.append(Layer(sizes[i], sizes[i + 1], "linear" if is_last else nonlin))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
