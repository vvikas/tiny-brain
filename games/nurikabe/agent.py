"""
games/nurikabe/agent.py

NurikabeAgent — wraps a tiny-brain MLP to solve Nurikabe puzzles iteratively.

Architecture: 50 → 256 → 256 → 25
  Input  : 50 floats = [clue_channel (25)] + [state_channel (25)]
             clue[i]  = size/5.0  for clue cells, 0 elsewhere
             state[i] = +1 (white locked), -1 (black locked), 0 (unknown)
  Output : 25 logits → sigmoid → P(cell is white/island)

Iterative solving:
  Each call to solve_iterative() locks one cell (most confident unknown),
  feeds the updated state back, and repeats until all cells are filled.
  This matches how the model was trained (partial-reveal training).
"""

import math
import pickle
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tinybrain.engine import Value
from tinybrain.nn import MLP
from games.nurikabe.puzzle import SIZE


def _sigmoid(x_float):
    """Numerically stable sigmoid for plain floats."""
    if x_float >= 0:
        return 1.0 / (1.0 + math.exp(-x_float))
    else:
        e = math.exp(x_float)
        return e / (1.0 + e)


def build_input(clues, state_dict, size=SIZE):
    """
    Build a 98-float input vector.

    clues      : dict {cell_index: island_size}
    state_dict : dict {cell_index: +1 (white) or -1 (black)}

    Returns list of 98 floats:
        [clue_channel (49)] + [state_channel (49)]
    """
    clue_ch  = [0.0] * size
    state_ch = [0.0] * size
    for ci, sz in clues.items():
        clue_ch[ci] = sz / 5.0
    for ci, val in state_dict.items():
        state_ch[ci] = float(val)
    return clue_ch + state_ch


class NurikabeAgent:
    """
    Inference wrapper around a tiny-brain MLP for iterative Nurikabe solving.

    Usage:
        agent = NurikabeAgent.load('agent.pkl')

        # Full iterative solve (returns 49-int solution)
        solution = agent.solve_iterative(clues)

        # One step (returns next cell + updated state + all probs)
        cell, value, conf, probs, state = agent.step(clues, state)
    """

    def __init__(self, layer_sizes=(50, 256, 256, 25)):
        self._layer_sizes = list(layer_sizes)
        n_in   = layer_sizes[0]
        n_outs = list(layer_sizes[1:])
        self.mlp = MLP(n_in, n_outs)

    # ── Core inference ────────────────────────────────────────────────────── #

    def forward(self, clues, state_dict):
        """
        Run one forward pass given current clues + locked cells.

        Returns list of 49 floats — P(white) per cell.
        """
        inp = build_input(clues, state_dict)
        return self._forward_raw(inp)

    def _forward_raw(self, inp):
        inputs_v = [Value(s) for s in inp]
        logits_v = self.mlp(inputs_v)
        if not isinstance(logits_v, list):
            logits_v = [logits_v]
        return [_sigmoid(v.data) for v in logits_v]

    # ── Iterative solving ─────────────────────────────────────────────────── #

    def step(self, clues, state_dict):
        """
        Run one step of iterative solving.

        Finds the most confident unknown cell, locks it, and returns:
          (cell_index, value, confidence, probs, updated_state_dict)

        confidence = |prob - 0.5|  in [0, 0.5]  (0.5 = maximally certain)

        If all cells are already locked, returns None.
        """
        unknown = [i for i in range(SIZE) if i not in state_dict]
        if not unknown:
            return None

        probs = self.forward(clues, state_dict)

        best_cell = max(unknown, key=lambda c: abs(probs[c] - 0.5))
        best_prob = probs[best_cell]
        value     = 1 if best_prob >= 0.5 else 0
        conf      = abs(best_prob - 0.5)

        new_state = dict(state_dict)
        new_state[best_cell] = 1 if value == 1 else -1

        return best_cell, value, conf, probs, new_state

    def solve_iterative(self, clues):
        """
        Run the full iterative solve from clues only.

        Returns list of 49 ints (0=black, 1=white).
        """
        # Clue cells are always white
        state = {ci: 1 for ci in clues}

        unknown = [i for i in range(SIZE) if i not in state]
        while unknown:
            result = self.step(clues, state)
            if result is None:
                break
            cell, value, conf, probs, state = result
            unknown = [i for i in range(SIZE) if i not in state]

        return [state.get(i, 0) for i in range(SIZE)]

    # ── Brain state for visualizer ────────────────────────────────────────── #

    def get_brain_state(self, clues, state_dict=None):
        """
        Run a forward pass and capture per-layer activations.

        state_dict: partial locked state (default: empty, clues only)

        Returns dict: inputs, hidden1, hidden2, logits, probs
        """
        if state_dict is None:
            state_dict = {ci: 1 for ci in clues}

        inp = build_input(clues, state_dict)
        inputs_v = [Value(s) for s in inp]

        x = inputs_v
        hidden_activations = []
        for layer in self.mlp.layers[:-1]:
            x = layer(x)
            hidden_activations.append([v.data for v in x])

        logits_v = self.mlp.layers[-1](x)
        logits   = [v.data for v in logits_v]
        probs    = [_sigmoid(l) for l in logits]

        result = {'inputs': inp, 'logits': logits, 'probs': probs}
        for i, acts in enumerate(hidden_activations):
            result[f'hidden{i + 1}'] = acts
        return result

    # ── Persistence ───────────────────────────────────────────────────────── #

    def save(self, path):
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
        agent = NurikabeAgent(tuple(data['layer_sizes']))
        for p, w in zip(agent.mlp.parameters(), data['weights']):
            p.data = w
        return agent
