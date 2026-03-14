"""
games/nurikabe/server.py

Flask web server for the Nurikabe iterative NN solver.

Endpoints:
  GET  /                     → game UI
  GET  /api/puzzle/new       → new puzzle + first NN forward pass
  POST /api/puzzle/step      → one iterative NN step
                               body: {"clues": {...}, "state": {cell: +1/-1, ...}}
                               returns: cell, value, confidence, probs, state
  POST /api/puzzle/solve     → run full iterative solve, return all steps
  POST /api/puzzle/check     → validate user solution
  GET  /api/brain            → raw activations for query clue vector

Run with:  python3 games/nurikabe/server.py
Then open: http://localhost:5001
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from flask import Flask, jsonify, request, render_template
from games.nurikabe.puzzle import (
    generate_puzzle, is_valid_solution, SIZE, ROWS, COLS,
    check_no_pools, check_river_connected, check_islands,
)
from games.nurikabe.agent import NurikabeAgent

app = Flask(__name__, template_folder='templates')

AGENT_PATH = os.path.join(os.path.dirname(__file__), 'agent.pkl')


# ── Load agent ────────────────────────────────────────────────────────────── #

def load_agent():
    if os.path.exists(AGENT_PATH):
        print(f"Loading trained Nurikabe agent from {AGENT_PATH}")
        return NurikabeAgent.load(AGENT_PATH)
    else:
        print("No trained agent found — using untrained weights.")
        print("Train one with:  python3 games/nurikabe/train_pytorch.py")
        return NurikabeAgent()


agent = load_agent()


# ── Routes ────────────────────────────────────────────────────────────────── #

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/puzzle/new')
def new_puzzle():
    """Generate a new puzzle and return the initial NN forward pass (no cells locked yet)."""
    clues, solution = generate_puzzle()
    clues_json = {str(k): v for k, v in clues.items()}

    # Initial state: clue cells locked as white
    init_state = {ci: 1 for ci in clues}
    brain = agent.get_brain_state(clues, init_state)

    return jsonify({
        'clues':    clues_json,
        'solution': solution,       # ground truth (shown after reveal)
        'brain':    brain,          # initial NN activations
        'rows':     ROWS,
        'cols':     COLS,
    })


@app.route('/api/puzzle/step', methods=['POST'])
def puzzle_step():
    """
    Run ONE step of iterative NN solving.

    Body:
        clues : {str(cell_index): island_size, ...}
        state : {str(cell_index): +1 or -1, ...}   ← already-locked cells

    Returns:
        cell       : int — cell index just locked
        value      : 0 or 1
        confidence : float in [0, 0.5]
        probs      : list of 49 floats (P(white) after this step's forward pass)
        state      : updated state dict (str keys for JSON)
        done       : bool — all cells are now locked
    """
    data  = request.get_json()
    clues = {int(k): int(v) for k, v in data['clues'].items()}
    state = {int(k): int(v) for k, v in data.get('state', {}).items()}

    result = agent.step(clues, state)
    if result is None:
        return jsonify({'done': True})

    cell, value, conf, probs, new_state = result
    done = all(i in new_state for i in range(SIZE))

    return jsonify({
        'cell':       cell,
        'value':      value,
        'confidence': round(conf, 4),
        'probs':      [round(p, 4) for p in probs],
        'state':      {str(k): v for k, v in new_state.items()},
        'done':       done,
    })


@app.route('/api/puzzle/solve', methods=['POST'])
def solve_puzzle():
    """
    Run the FULL iterative solve and return every step taken.

    Body:  {"clues": {str: int, ...}}

    Returns:
        steps : list of {cell, value, confidence, probs_snapshot}
                sorted in the order the NN locked them
        final_solution : list of 49 ints
    """
    data  = request.get_json()
    clues = {int(k): int(v) for k, v in data['clues'].items()}

    state   = {ci: 1 for ci in clues}
    steps   = []
    unknown = [i for i in range(SIZE) if i not in state]

    while unknown:
        result = agent.step(clues, state)
        if result is None:
            break
        cell, value, conf, probs, state = result
        steps.append({
            'cell':       cell,
            'value':      value,
            'confidence': round(conf, 4),
        })
        unknown = [i for i in range(SIZE) if i not in state]

    final = [state.get(i, 0) for i in range(SIZE)]
    # Force clue cells white in output
    for ci in clues:
        final[ci] = 1

    return jsonify({
        'steps':          steps,
        'final_solution': final,
    })


@app.route('/api/puzzle/check', methods=['POST'])
def check_solution():
    """Validate a user-drawn solution against Nurikabe rules."""
    data     = request.get_json()
    clues    = {int(k): int(v) for k, v in data['clues'].items()}
    solution = data.get('solution', [])

    if len(solution) != SIZE:
        return jsonify({'error': f'Solution must have {SIZE} cells'}), 400

    return jsonify({
        'valid':           is_valid_solution(solution, clues),
        'no_pools':        check_no_pools(solution),
        'river_connected': check_river_connected(solution),
        'islands_ok':      check_islands(solution, clues),
    })


@app.route('/api/brain')
def brain_state():
    """Return NN activations for a clue vector given as query parameter."""
    raw = request.args.get('clues', '')
    try:
        vals = [float(x) for x in raw.split(',')]
        if len(vals) != SIZE:
            raise ValueError
    except (ValueError, AttributeError):
        return jsonify({'error': f'Provide {SIZE} comma-separated values'}), 400

    clues = {i: round(v * 7) for i, v in enumerate(vals) if v > 0.01}
    brain = agent.get_brain_state(clues)
    return jsonify(brain)


# ── Main ──────────────────────────────────────────────────────────────────── #

if __name__ == '__main__':
    print()
    print("  tiny-brain Nurikabe — Iterative NN Solver")
    print("  Open http://localhost:5001 in your browser")
    print()
    app.run(debug=False, port=5001)
