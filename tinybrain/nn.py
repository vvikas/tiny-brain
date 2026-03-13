"""
tinybrain/nn.py

Neural network building blocks: Neuron, Layer, MLP.
All built on top of Value from engine.py — no numpy, no PyTorch.

Inspired by Andrej Karpathy's micrograd (https://youtu.be/VMj-3S1tku0).
"""

import random
from tinybrain.engine import Value


class Neuron:
    """
    A single artificial neuron.

    Does: output = activation( sum(w_i * x_i) + b )

    Parameters:
        n_in   : number of inputs
        nonlin : if True, apply ReLU activation; if False, output is linear
                 (the last layer of an MLP is always linear so gradients
                 aren't clamped before the loss function sees them)
    """

    def __init__(self, n_in, nonlin=True):
        # Kaiming / He initialisation: scale = sqrt(2 / n_in)
        # This keeps activation variance roughly constant through a ReLU network,
        # preventing outputs from exploding or vanishing as the network deepens.
        scale = (2.0 / n_in) ** 0.5
        self.w = [Value(random.gauss(0.0, scale)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        # x is a list of numbers or Value objects
        # sum() with Value(0) start so the whole expression stays in the graph
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        kind = 'ReLU' if self.nonlin else 'Linear'
        return f"{kind}Neuron({len(self.w)} inputs)"


class Layer:
    """
    A layer of independent neurons all reading the same input.

    Each neuron produces one output, so a Layer(n_in, n_out) maps
    a vector of length n_in → a vector of length n_out.
    """

    def __init__(self, n_in, n_out, **kwargs):
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer({self.neurons})"


class MLP:
    """
    Multi-Layer Perceptron.

    Usage:
        model = MLP(2, [16, 16, 1])
        # 2 inputs → hidden layer of 16 → hidden layer of 16 → 1 output

    The last layer is always linear (no activation) so the network can
    output any real number before a loss function or softmax is applied.
    All other layers use ReLU.

    Key methods:
        model(x)          — forward pass, returns output Value(s)
        model.parameters()— flat list of all weight/bias Values
        model.zero_grad() — reset all gradients to 0 before a new backward pass
    """

    def __init__(self, n_in, n_outs):
        sizes = [n_in] + n_outs
        self.layers = [
            # nonlin=True for all layers except the last
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(n_outs) - 1))
            for i in range(len(n_outs))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        # Unwrap single-element output lists for convenience:
        # MLP(2,[16,16,1])(x) returns a single Value, not [Value]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        """
        Reset all parameter gradients to zero.

        Must be called before every .backward() call, otherwise gradients
        accumulate across steps (because _backward closures use +=).
        """
        for p in self.parameters():
            p.grad = 0.0

    def __repr__(self):
        return f"MLP({self.layers})"
