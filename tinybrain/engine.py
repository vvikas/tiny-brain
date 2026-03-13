"""
tinybrain/engine.py

Scalar-valued autograd engine.
Inspired by Andrej Karpathy's micrograd (https://youtu.be/VMj-3S1tku0).

Every Value wraps a single Python float (64-bit double) and knows how to
compute the gradient of whatever scalar loss you eventually build from it.
"""

import sys
import math

sys.setrecursionlimit(10000)


class Value:
    """
    A node in a computation graph.

    Stores:
        data  : the actual number this node holds
        grad  : d(loss)/d(this), filled in by .backward()
        label : optional name for pretty-printing / graph viz
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        # _backward is a closure set by each op that knows how to
        # push gradients back to this node's inputs.
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op        # what created this node, e.g. '+', '*', 'relu'
        self.label = label    # human-readable name (optional)

    # ------------------------------------------------------------------ #
    #  Primitive ops — each one builds a new Value and defines its        #
    #  _backward closure using the chain rule.                            #
    # ------------------------------------------------------------------ #

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1  =>  both inputs get out.grad
            self.grad  += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a*b)/da = b,  d(a*b)/db = a
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        # exponent must be a plain Python int or float — not a Value
        assert isinstance(exponent, (int, float)), \
            "__pow__ only supports int/float exponents, not Value"
        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """Rectified Linear Unit: max(0, x). The most common hidden-layer activation."""
        out = Value(max(0.0, self.data), (self,), 'relu')

        def _backward():
            # Derivative is 1 if x > 0, else 0
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """
        Hyperbolic tangent: squashes any real number to (-1, 1).
        Used in Karpathy's original neuron demo and in sigmoid-like gates.
        """
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """e^x — used inside numerically-stable softmax for game agents."""
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            # d(e^x)/dx = e^x
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        """
        Natural logarithm — needed for policy gradient (REINFORCE) in game agents:
            loss = -log(prob_of_chosen_action) * reward
        The epsilon prevents log(0) which would give -inf.
        """
        eps = 1e-8
        out = Value(math.log(self.data + eps), (self,), 'log')

        def _backward():
            # d(ln(x))/dx = 1/x
            self.grad += (1.0 / (self.data + eps)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    #  Derived ops — built from the primitives above so we get their      #
    #  backward passes for free.                                          #
    # ------------------------------------------------------------------ #

    def __neg__(self):             return self * -1
    def __sub__(self, other):      return self + (-other)
    def __truediv__(self, other):  return self * (other ** -1)

    # "Reflected" versions so  `3 + Value(2)`  works the same as  `Value(2) + 3`
    def __radd__(self, other):     return self + other
    def __rmul__(self, other):     return self * other
    def __rsub__(self, other):     return Value(other) - self
    def __rtruediv__(self, other): return Value(other) / self

    # ------------------------------------------------------------------ #
    #  Backpropagation                                                    #
    # ------------------------------------------------------------------ #

    def backward(self):
        """
        Compute gradients for every Value that contributed to self.

        Algorithm:
          1. Topological sort — visit each node only after all nodes that
             depend on it have been visited (i.e. children before parents
             in the graph sense, which means output-to-input ordering).
          2. Seed this node's gradient with 1.0  (d(self)/d(self) = 1).
          3. Walk the sorted list in reverse, calling each node's
             _backward() closure to distribute gradient to its inputs.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
