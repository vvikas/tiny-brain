"""
games/tictactoe/server.py

Flask web server for the TicTacToe brain visualizer.

Endpoints:
  GET  /              → serve the game UI (index.html)
  GET  /api/new_game  → reset board, return initial state
  POST /api/move      → human plays a move, AI responds
                        body: {"pos": 0-8}
                        returns: game state + brain state for visualizer
  GET  /api/brain     → current brain state (for a hypothetical board)
                        query param: ?board=0,1,-1,0,0,0,0,0,0

Run with:  python games/tictactoe/server.py
Then open: http://localhost:5000
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from flask import Flask, jsonify, request, render_template, send_from_directory
from games.tictactoe.game import TicTacToe
from games.tictactoe.agent import NNAgent

app = Flask(__name__, template_folder='templates')

AGENT_PATH = os.path.join(os.path.dirname(__file__), 'agent.pkl')

# ── Load agent ────────────────────────────────────────────────────────── #

def load_agent():
    if os.path.exists(AGENT_PATH):
        print(f"Loading trained agent from {AGENT_PATH}")
        return NNAgent.load(AGENT_PATH)
    else:
        print("No trained agent found — using untrained agent.")
        print(f"Train one with:  python games/tictactoe/train.py")
        return NNAgent()

agent = load_agent()
game  = TicTacToe()


# ── Helper ────────────────────────────────────────────────────────────── #

def game_response(extra=None):
    """Build the standard JSON response with board + brain state."""
    state      = game.get_state()
    valid      = game.get_valid_moves()
    brain      = agent.get_brain_state(state, valid)

    payload = {
        "board":        game.board,
        "current_player": game.current_player,
        "valid_moves":  valid,
        "brain":        brain,
    }
    if extra:
        payload.update(extra)
    return jsonify(payload)


# ── Routes ────────────────────────────────────────────────────────────── #

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/new_game')
def new_game():
    game.reset()
    return game_response({"message": "New game started. You are O (-1). AI is X (1)."})


@app.route('/api/move', methods=['POST'])
def move():
    data = request.get_json()
    pos  = int(data.get('pos', -1))

    if pos not in game.get_valid_moves():
        return jsonify({"error": "Invalid move"}), 400

    # Human plays as O (-1).
    # The game starts with current_player = 1 (X = AI).
    # We let the AI go first if it's X's turn, then human goes.
    # But the human sends a move, so it must be their turn (O = -1).

    # Human move
    result = game.make_move(pos)
    if result != 'ongoing':
        winner = "draw" if result == 'draw' else ("O" if game.current_player == -1 else "X")
        return game_response({"result": result, "winner": winner, "ai_move": None})

    # AI move (X = 1)
    state = game.get_state()
    valid = game.get_valid_moves()
    if not valid:
        return game_response({"result": "draw", "winner": "draw", "ai_move": None})

    ai_action, _ = agent.select_action(state, valid, training=False)
    result = game.make_move(ai_action)

    if result != 'ongoing':
        winner = "draw" if result == 'draw' else ("X" if game.current_player == 1 else "O")
        return game_response({"result": result, "winner": winner, "ai_move": ai_action})

    return game_response({"result": "ongoing", "winner": None, "ai_move": ai_action})


@app.route('/api/ai_first', methods=['POST'])
def ai_first():
    """Let the AI make the first move (so human plays as O)."""
    if game.current_player != 1:
        return jsonify({"error": "Not AI's turn"}), 400

    state = game.get_state()
    valid = game.get_valid_moves()
    ai_action, _ = agent.select_action(state, valid, training=False)
    result = game.make_move(ai_action)

    return game_response({"result": result, "ai_move": ai_action})


if __name__ == '__main__':
    print()
    print("  tiny-brain TicTacToe — Brain Visualizer")
    print("  Open http://localhost:5000 in your browser")
    print()
    app.run(debug=False, port=5000)
