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

HUMAN_PLAYER = -1   # human is always O; AI is always X


# ── Helper ────────────────────────────────────────────────────────────── #

def game_response(brain=None, extra=None):
    """Build the standard JSON response with board state.
    brain is passed explicitly — None when it's the human's turn (clears heatmap).
    """
    payload = {
        "board":          game.board,
        "current_player": game.current_player,
        "valid_moves":    game.get_valid_moves(),
        "brain":          brain,
        "human_player":   HUMAN_PLAYER,
    }
    if extra:
        payload.update(extra)
    return jsonify(payload)


def ai_move():
    """Make one AI move and return (result, action, brain_before_move).
    Brain state is captured BEFORE placing so it shows what the AI evaluated
    when making its decision (not what it thinks about the human's next turn).

    The board is passed from the AI's perspective (own pieces = +1, opponent = -1)
    to match how the agent was trained (perspective-flipped inputs).
    """
    current = game.current_player
    state_abs = game.get_state()          # absolute: 1=X, -1=O
    valid = game.get_valid_moves()
    # Flip to agent's own perspective so it sees itself as "player 1"
    state_persp = [s * current for s in state_abs]
    brain_before = agent.get_brain_state(state_persp, valid)
    action, _ = agent.select_action(state_persp, valid, training=False)
    result = game.make_move(action)
    return result, action, brain_before


# ── Routes ────────────────────────────────────────────────────────────── #

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/new_game')
def new_game():
    # AI always goes first as X; human is always O
    game.reset()
    result, action, brain_before = ai_move()
    return game_response(brain=brain_before, extra={"result": result, "ai_move": action})


@app.route('/api/move', methods=['POST'])
def move():
    data = request.get_json()
    pos  = int(data.get('pos', -1))

    if pos not in game.get_valid_moves():
        return jsonify({"error": "Invalid move"}), 400

    if game.current_player != HUMAN_PLAYER:
        return jsonify({"error": "Not your turn"}), 400

    # Human move
    result = game.make_move(pos)
    if result != 'ongoing':
        winner = "draw" if result == 'draw' else "You win!"
        return game_response(brain=None, extra={"result": result, "winner": winner, "ai_move": None})

    # AI's turn — capture brain before placing
    result, action, brain_before = ai_move()
    if result != 'ongoing':
        winner = "draw" if result == 'draw' else "AI wins!"
        return game_response(brain=brain_before, extra={"result": result, "winner": winner, "ai_move": action})

    return game_response(brain=brain_before, extra={"result": "ongoing", "winner": None, "ai_move": action})


if __name__ == '__main__':
    print()
    print("  tiny-brain TicTacToe — Brain Visualizer")
    print("  Open http://localhost:5000 in your browser")
    print()
    app.run(debug=False, port=5000)
