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

# human_player: 1 = human plays X (goes first), -1 = human plays O (AI goes first)
human_player = -1


# ── Helper ────────────────────────────────────────────────────────────── #

def game_response(extra=None):
    """Build the standard JSON response with board + brain state."""
    state = game.get_state()
    valid = game.get_valid_moves()
    brain = agent.get_brain_state(state, valid)

    payload = {
        "board":          game.board,
        "current_player": game.current_player,
        "valid_moves":    valid,
        "brain":          brain,
        "human_player":   human_player,
    }
    if extra:
        payload.update(extra)
    return jsonify(payload)


def ai_move():
    """Make one AI move and return (result, ai_action)."""
    state = game.get_state()
    valid = game.get_valid_moves()
    action, _ = agent.select_action(state, valid, training=False)
    result = game.make_move(action)
    return result, action


# ── Routes ────────────────────────────────────────────────────────────── #

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/new_game')
def new_game():
    global human_player
    # ?human=X → human goes first as X; anything else → AI goes first
    human_player = 1 if request.args.get('human') == 'X' else -1

    game.reset()

    if human_player == -1:
        # AI (X=1) plays first
        result, action = ai_move()
        return game_response({"result": result, "ai_move": action})
    else:
        # Human (X=1) plays first — board is empty, human's turn
        return game_response({"result": "ongoing", "ai_move": None})


@app.route('/api/move', methods=['POST'])
def move():
    data = request.get_json()
    pos  = int(data.get('pos', -1))

    if pos not in game.get_valid_moves():
        return jsonify({"error": "Invalid move"}), 400

    if game.current_player != human_player:
        return jsonify({"error": "Not your turn"}), 400

    # Human move
    result = game.make_move(pos)
    if result != 'ongoing':
        winner = "draw" if result == 'draw' else "You win!"
        return game_response({"result": result, "winner": winner, "ai_move": None})

    # AI's turn
    result, action = ai_move()
    if result != 'ongoing':
        winner = "draw" if result == 'draw' else "AI wins!"
        return game_response({"result": result, "winner": winner, "ai_move": action})

    return game_response({"result": "ongoing", "winner": None, "ai_move": action})


if __name__ == '__main__':
    print()
    print("  tiny-brain TicTacToe — Brain Visualizer")
    print("  Open http://localhost:5000 in your browser")
    print()
    app.run(debug=False, port=5000)
