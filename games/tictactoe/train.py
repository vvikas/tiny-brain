"""
games/tictactoe/train.py

Train the TicTacToe agent using REINFORCE (policy gradient).

Key improvements over the naive version:
  1. Perspective-flipped board: agent always receives its own pieces as +1
     and opponent's as -1, regardless of whether it's playing as X or O.
     This means one set of weights works for both roles.
  2. Tactical reward shaping: extra reward/penalty when the agent correctly
     or incorrectly handles an immediate win or block opportunity.
     REINFORCE's sparse end-game reward alone struggles to teach tactics.
  3. Phase 1 alternates X/O: agent trains as both X and O vs random.
  4. More episodes: 10 000 vs random + 20 000 self-play.

How REINFORCE works:
  - Play a full game, collecting (log_prob, reward) pairs for every agent move.
  - At the end, assign the game outcome as reward (+1 win, -1 loss, 0 draw).
  - Add a tactical shaping bonus/penalty to each step.
  - loss = -sum(log_prob * (reward + shaping - baseline))
  - loss.backward() flows gradients through the network.
  - SGD nudges weights to make winning (and tactically correct) moves more likely.
"""

import random
import sys
import os
import time
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.tictactoe.game import TicTacToe
from games.tictactoe.agent import NNAgent


AGENT_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'agent.pkl')


# ── Tactical helpers ──────────────────────────────────────────────────────── #

_WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],   # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],   # columns
    [0, 4, 8], [2, 4, 6],              # diagonals
]


def _immediate_win(board, valid_moves, player):
    """Return a move that wins immediately for `player`, or None."""
    for m in valid_moves:
        test = list(board)
        test[m] = player
        if any(all(test[i] == player for i in line) for line in _WIN_LINES):
            return m
    return None


def _tactical_shaping(board, valid_moves, player, chosen):
    """
    Extra reward/penalty for tactical win/block decisions.

    Provides dense reward signal so the agent learns to:
      - Always take an immediate win (+0.5 if correct, -0.6 if missed)
      - Always block an opponent's immediate win (+0.3 if blocked, -0.5 if missed)

    These bonuses are added on top of the game-end reward before backprop.
    """
    win_move = _immediate_win(board, valid_moves, player)
    if win_move is not None:
        return 0.5 if chosen == win_move else -0.6

    block_move = _immediate_win(board, valid_moves, -player)
    if block_move is not None:
        return 0.3 if chosen == block_move else -0.5

    return 0.0


def board_perspective(board, player):
    """
    Return the board from `player`'s point of view:
        own pieces  → +1
        opponent    → -1
        empty       →  0

    This lets a single set of weights serve both X and O roles.
    """
    return [float(c * player) for c in board]


# ── Episode functions ─────────────────────────────────────────────────────── #

def play_episode_vs_random(agent, game, lr, mean_reward, agent_player=1):
    """
    One episode: agent plays as `agent_player` vs a random opponent.
    Uses perspective-flipped board so agent always sees itself as player 1.
    Returns (reward, updated_mean_reward).
    """
    game.reset()
    trajectory = []   # list of [log_prob Value, final_reward, tactical_shaping]

    while True:
        valid = game.get_valid_moves()

        if game.current_player == agent_player:
            # Give agent its own perspective: own = +1, opp = -1
            persp = board_perspective(game.board, agent_player)
            action, log_prob = agent.select_action(persp, valid, training=True)
            shaping = _tactical_shaping(game.board, valid, agent_player, action)
            trajectory.append([log_prob, None, shaping])
        else:
            # Random opponent — no gradient needed
            action = random.choice(valid)

        result = game.make_move(action)

        if result != 'ongoing':
            if result == 'win':
                reward = 1.0 if game.current_player == agent_player else -1.0
            else:
                reward = 0.0
            for step in trajectory:
                step[1] = reward
            break

    # Update running mean baseline (exponential moving average)
    alpha = 0.05
    mean_reward = (1 - alpha) * mean_reward + alpha * reward

    # REINFORCE update with tactical shaping
    if trajectory:
        agent.mlp.zero_grad()
        loss = sum(
            -lp * (r + shaping - mean_reward)
            for lp, r, shaping in trajectory
            if r is not None
        )
        loss.backward()
        for p in agent.mlp.parameters():
            p.data -= lr * p.grad

    return reward, mean_reward


def play_episode_vs_frozen(agent, frozen, game, lr, mean_reward, agent_player):
    """
    One symmetric self-play episode: `agent` plays as `agent_player` and learns;
    a frozen snapshot of the agent plays the other side (no gradient, no update).

    Structurally identical to play_episode_vs_random — same single-agent design —
    but uses a smarter, self-trained opponent instead of a random one.
    Alternating agent_player each episode ensures X and O training are symmetric.
    """
    game.reset()
    trajectory = []

    while True:
        valid = game.get_valid_moves()

        if game.current_player == agent_player:
            persp   = board_perspective(game.board, agent_player)
            action, log_prob = agent.select_action(persp, valid, training=True)
            shaping = _tactical_shaping(game.board, valid, agent_player, action)
            trajectory.append([log_prob, None, shaping])
        else:
            opp   = -agent_player
            persp = board_perspective(game.board, opp)
            action, _ = frozen.select_action(persp, valid, training=False)

        result = game.make_move(action)

        if result != 'ongoing':
            if result == 'win':
                reward = 1.0 if game.current_player == agent_player else -1.0
            else:
                reward = 0.0
            for step in trajectory:
                step[1] = reward
            break

    alpha = 0.05
    mean_reward = (1 - alpha) * mean_reward + alpha * reward

    if trajectory:
        agent.mlp.zero_grad()
        loss = sum(
            -lp * (r + shaping - mean_reward)
            for lp, r, shaping in trajectory
            if r is not None
        )
        loss.backward()
        for p in agent.mlp.parameters():
            p.data -= lr * p.grad

    return reward, mean_reward


# ── Evaluation ────────────────────────────────────────────────────────────── #

def eval_vs_random(agent, n_games=200):
    """
    Measure agent win/draw/loss rates against a random opponent.
    Alternates playing as X and O so we test both roles equally.
    """
    game = TicTacToe()
    wins = draws = losses = 0
    for ep in range(n_games):
        game.reset()
        agent_player = 1 if (ep % 2 == 0) else -1
        while True:
            valid = game.get_valid_moves()
            if game.current_player == agent_player:
                persp = board_perspective(game.board, agent_player)
                action, _ = agent.select_action(persp, valid, training=False)
            else:
                action = random.choice(valid)
            result = game.make_move(action)
            if result != 'ongoing':
                if result == 'win':
                    if game.current_player == agent_player:
                        wins += 1
                    else:
                        losses += 1
                else:
                    draws += 1
                break
    return wins / n_games, draws / n_games, losses / n_games


# ── Main training loop ────────────────────────────────────────────────────── #

def train(
    phase1_episodes=10000,
    phase2_episodes=20000,
    lr=0.01,
    lr_phase2=0.002,   # lower LR for Phase 2: fine-tune without overwriting Phase 1
    eval_every=1000,
    save_path=AGENT_SAVE_PATH,
):
    print("=" * 60)
    print("tiny-brain TicTacToe — REINFORCE Training")
    print("=" * 60)
    print(f"Phase 1: {phase1_episodes} episodes vs random (alternates X/O), lr={lr}")
    print(f"Phase 2: {phase2_episodes} episodes of symmetric self-play (alternates X/O), lr={lr_phase2}")
    print("Enhancements: perspective-flipped board + tactical reward shaping")
    print()

    agent = NNAgent()
    game  = TicTacToe()
    mean_reward = 0.0
    win_history = []

    # ── Phase 1: vs random ────────────────────────────────────────────── #
    print("Phase 1: Training vs random opponent (alternating X and O)...")
    t0 = time.time()

    for ep in range(1, phase1_episodes + 1):
        # Alternate: odd episodes as X, even episodes as O
        agent_player = 1 if (ep % 2 == 1) else -1
        reward, mean_reward = play_episode_vs_random(
            agent, game, lr, mean_reward, agent_player
        )
        win_history.append(1 if reward > 0 else 0)

        if ep % eval_every == 0:
            recent = sum(win_history[-eval_every:]) / eval_every
            w, d, l = eval_vs_random(agent, 200)
            elapsed = time.time() - t0
            print(f"  ep {ep:5d}: recent_win={recent:.1%}  "
                  f"eval W={w:.1%} D={d:.1%} L={l:.1%}  "
                  f"({elapsed:.0f}s)")

    print()
    w, d, l = eval_vs_random(agent, 500)
    print(f"End of Phase 1 — W={w:.1%} D={d:.1%} L={l:.1%}")
    print()

    # ── Phase 2: symmetric self-play ──────────────────────────────────── #
    print("Phase 2: Symmetric self-play (agent alternates X and O vs frozen self)...")
    frozen = copy.deepcopy(agent)
    t0 = time.time()

    for ep in range(1, phase2_episodes + 1):
        # Alternate roles each episode — same single-agent design as Phase 1
        agent_player = 1 if (ep % 2 == 1) else -1
        reward, mean_reward = play_episode_vs_frozen(
            agent, frozen, game, lr_phase2, mean_reward, agent_player
        )
        win_history.append(1 if reward > 0 else 0)

        # Periodically sharpen the frozen opponent
        if ep % 2000 == 0:
            frozen = copy.deepcopy(agent)

        if ep % eval_every == 0:
            recent = sum(win_history[-eval_every:]) / eval_every
            w, d, l = eval_vs_random(agent, 200)
            elapsed = time.time() - t0
            print(f"  ep {ep:5d}: recent_win={recent:.1%}  "
                  f"eval W={w:.1%} D={d:.1%} L={l:.1%}  ({elapsed:.0f}s)")

    print()
    w, d, l = eval_vs_random(agent, 500)
    print(f"End of Phase 2 — W={w:.1%} D={d:.1%} L={l:.1%}")
    print()

    # ── Save ───────────────────────────────────────────────────────────── #
    agent.save(save_path)
    print(f"Agent saved to {save_path}")
    print("Run  python games/tictactoe/server.py  to play against it in the browser.")

    return agent


if __name__ == '__main__':
    train()
