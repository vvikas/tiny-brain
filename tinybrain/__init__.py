"""
tiny-brain: a from-scratch neural network library.

Inspired by Andrej Karpathy's micrograd (https://youtu.be/VMj-3S1tku0).
Pure Python, zero dependencies.
"""

from tinybrain.engine import Value
from tinybrain.nn import Neuron, Layer, MLP

__all__ = ["Value", "Neuron", "Layer", "MLP"]
