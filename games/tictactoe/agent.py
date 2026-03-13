"""
games/tictactoe/agent.py

NNAgent — wraps an MLP to play TicTacToe.

Key responsibilities:
  1. Forward pass: board state → 9 logits → softmax probabilities
  2. Action selection: sample (training) or argmax (greedy/play mode)
  3. Brain state export: activation data at each layer for the visualizer

The `log` op is used here for REINFORCE:
    loss = -log(prob_of_chosen_action) * reward
"""

import math
import random
import pickle
import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tinybrain.engine import Value
from tinybrain.nn import MLP


def softmax(logits):
    """
    Numerically stable softmax over a list of Value objects.

    Why stable? Without subtracting the max, exp(large_number) overflows
    to inf, and inf/inf = nan, killing all gradients permanently.

    We subtract max_val (a plain float, not a Value) so the gradient
    still flows correctly through the Value graph.
    """
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    # sum all exps; use the first as the starting value for sum()
    total = sum(exps[1:], exps[0])
    return [e / total for e in exps]


def sample_from_probs(probs, valid_moves):
    """
    Sample one action from the probability distribution,
    considering only valid (unoccupied) moves.
    """
    r = random.random()
    cumulative = 0.0
    for i in valid_moves:
        cumulative += probs[i].data
        if r <= cumulative:
            return i
    return valid_moves[-1]   # floating-point edge case fallback


class NNAgent:
    """
    Neural network TicTacToe agent.

    Architecture: 9 → 32 → 32 → 9
      Input  : 9 board cells (0=empty, 1=X, -1=O)
      Hidden : two ReLU layers of 32 neurons each
      Output : 9 logits, one per board cell (higher = more preferred)
    """

    def __init__(self, layer_sizes=(9, 32, 32, 9)):
        self._layer_sizes = list(layer_sizes)
        n_in = layer_sizes[0]
        n_outs = list(layer_sizes[1:])
        self.mlp = MLP(n_in, n_outs)

    def _forward(self, state):
        """
        Run the board state through the network.
        Returns 9 Value logits (raw scores before softmax).
        """
        inputs = [Value(s) for s in state]
        return self.mlp(inputs)   # list of 9 Values

    def select_action(self, state, valid_moves, training=True):
        """
        Choose a move given the current board state.

        Args:
            state       : list of 9 floats (board)
            valid_moves : list of int indices of empty cells
            training    : if True, sample from distribution (explores);
                          if False, take argmax (deterministic/greedy)

        Returns:
            (action, log_prob)
            action   : int 0-8, the chosen cell
            log_prob : Value — log probability of the chosen action.
                       This is the handle we call .backward() on during training.
        """
        logits = self._forward(state)

        # Mask invalid moves by setting their logit to a large negative number.
        # We use -1e9 instead of -inf because:
        #   exp(-inf) = 0  is fine, but
        #   (-inf) - max_val = -inf - (-inf) = nan  breaks stable softmax.
        masked_logits = []
        for i, logit in enumerate(logits):
            if i in valid_moves:
                masked_logits.append(logit)
            else:
                masked_logits.append(Value(-1e9))

        probs = softmax(masked_logits)

        if training:
            action = sample_from_probs(probs, valid_moves)
        else:
            action = max(valid_moves, key=lambda i: probs[i].data)

        log_prob = probs[action].log()
        return action, log_prob

    def get_brain_state(self, state, valid_moves=None):
        """
        Run a forward pass and capture activations at every layer.
        This data is sent to the browser for the neural network visualizer.

        Returns a dict:
            inputs   : 9 floats (the board values fed in)
            hidden1  : 32 floats (post-ReLU activations of layer 1)
            hidden2  : 32 floats (post-ReLU activations of layer 2)
            logits   : 9 floats (raw output scores)
            probs    : 9 floats (softmax probabilities, masked for valid moves)
        """
        if valid_moves is None:
            valid_moves = [i for i, s in enumerate(state) if s == 0.0]

        inputs_v = [Value(s) for s in state]

        # Manually run through each layer to capture intermediate activations
        x = inputs_v
        hidden_activations = []
        for layer in self.mlp.layers[:-1]:    # all but last
            x = layer(x)
            hidden_activations.append([v.data for v in x])

        # Last layer (linear, no activation)
        logits_v = self.mlp.layers[-1](x)
        logits = [v.data for v in logits_v]

        # Softmax with masking (plain float math — no gradient tracking needed)
        masked = [l if i in valid_moves else -1e9 for i, l in enumerate(logits)]
        max_m = max(masked)
        exps = [math.exp(m - max_m) for m in masked]
        total = sum(exps)
        probs = [e / total for e in exps]

        result = {
            "inputs":  [float(s) for s in state],
            "logits":  logits,
            "probs":   probs,
        }
        for idx, acts in enumerate(hidden_activations):
            result[f"hidden{idx + 1}"] = acts

        return result

    def save(self, path):
        """
        Save only the weight values (plain floats) — not the computation graph.
        Value closures (lambdas) can't be pickled, so we extract just the numbers.
        """
        data = {
            'layer_sizes': self._layer_sizes,
            'weights': [p.data for p in self.mlp.parameters()],
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        agent = NNAgent(tuple(data['layer_sizes']))
        for p, w in zip(agent.mlp.parameters(), data['weights']):
            p.data = w
        return agent
