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


def play_episode_self_play(agent_x, agent_o, game, lr_x, lr_o, mean_x, mean_o):
    """
    One self-play episode: agent_x (X=1) vs agent_o (O=-1).
    Both agents receive the board from their own perspective.
    Returns (reward_x, reward_o, updated means).
    """
    game.reset()
    traj_x, traj_o = [], []

    while True:
        valid = game.get_valid_moves()

        if game.current_player == 1:
            persp = board_perspective(game.board, 1)    # X's view: no sign change
            action, log_prob = agent_x.select_action(persp, valid, training=True)
            shaping = _tactical_shaping(game.board, valid, 1, action)
            traj_x.append([log_prob, None, shaping])
        else:
            persp = board_perspective(game.board, -1)   # O's view: flip signs
            action, log_prob = agent_o.select_action(persp, valid, training=True)
            shaping = _tactical_shaping(game.board, valid, -1, action)
            traj_o.append([log_prob, None, shaping])

        result = game.make_move(action)

        if result != 'ongoing':
            winner = game.current_player
            if result == 'win':
                reward_x =  1.0 if winner == 1 else -1.0
                reward_o = -1.0 if winner == 1 else  1.0
            else:
                reward_x = reward_o = 0.0

            for step in traj_x:
                step[1] = reward_x
            for step in traj_o:
                step[1] = reward_o
            break

    alpha = 0.05
    mean_x = (1 - alpha) * mean_x + alpha * reward_x
    mean_o = (1 - alpha) * mean_o + alpha * reward_o

    def update(agent, traj, mean, lr):
        if traj:
            agent.mlp.zero_grad()
            loss = sum(-lp * (r + shaping - mean) for lp, r, shaping in traj if r is not None)
            loss.backward()
            for p in agent.mlp.parameters():
                p.data -= lr * p.grad

    update(agent_x, traj_x, mean_x, lr_x)
    update(agent_o, traj_o, mean_o, lr_o)

    return reward_x, reward_o, mean_x, mean_o


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
    eval_every=1000,
    save_path=AGENT_SAVE_PATH,
):
    print("=" * 60)
    print("tiny-brain TicTacToe — REINFORCE Training")
    print("=" * 60)
    print(f"Phase 1: {phase1_episodes} episodes vs random (alternates X/O)")
    print(f"Phase 2: {phase2_episodes} episodes of self-play")
    print(f"Learning rate: {lr}")
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

    # ── Phase 2: self-play ─────────────────────────────────────────────── #
    print("Phase 2: Self-play training (both agents use perspective flip)...")
    agent_o = copy.deepcopy(agent)

    mean_x = mean_o = 0.0
    t0 = time.time()

    for ep in range(1, phase2_episodes + 1):
        rx, ro, mean_x, mean_o = play_episode_self_play(
            agent, agent_o, game, lr, lr, mean_x, mean_o
        )

        # Periodically update the opponent with the latest weights
        if ep % 2000 == 0:
            agent_o = copy.deepcopy(agent)

        if ep % eval_every == 0:
            w, d, l = eval_vs_random(agent, 200)
            elapsed = time.time() - t0
            print(f"  ep {ep:5d}: eval vs random  W={w:.1%} D={d:.1%} L={l:.1%}  "
                  f"({elapsed:.0f}s)")

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
