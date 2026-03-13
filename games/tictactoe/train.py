"""
games/tictactoe/train.py

Train the TicTacToe agent using REINFORCE (policy gradient).

How REINFORCE works:
  - Play a full game (an "episode"), collecting (log_prob, reward) pairs
    for every move the agent made.
  - At the end, assign the game outcome as reward (+1 win, -1 loss, 0 draw).
  - loss = -sum(log_prob * advantage)  where advantage = reward - baseline
  - loss.backward() flows gradients all the way through the network.
  - SGD update: nudge weights in the direction that makes winning moves
    more likely and losing moves less likely.

Why a baseline?
  Without a baseline, REINFORCE has very high variance — the gradient
  estimates are noisy and training is slow. Subtracting the running
  average reward ("advantage = reward - mean_reward") doesn't change
  the expected gradient but dramatically reduces variance.

Training phases:
  Phase 1 (vs random)  : Agent (X) plays random opponent (O).
                         Gives a clear win-signal early on.
  Phase 2 (self-play)  : Two separate agents play each other.
                         Forces learning of counter-strategies.
"""

import random
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.tictactoe.game import TicTacToe
from games.tictactoe.agent import NNAgent


AGENT_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'agent.pkl')


def play_episode_vs_random(agent, game, lr, mean_reward):
    """
    One episode: agent (X=1) vs random opponent (O=-1).
    Returns (reward, updated_mean_reward).
    """
    game.reset()
    trajectory = []   # list of (log_prob Value, reward placeholder)

    while True:
        state = game.get_state()
        valid = game.get_valid_moves()

        if game.current_player == 1:
            # Agent's turn — track log_prob for REINFORCE
            action, log_prob = agent.select_action(state, valid, training=True)
            trajectory.append([log_prob, None])
        else:
            # Random opponent — no gradient needed
            action = random.choice(valid)

        result = game.make_move(action)

        if result != 'ongoing':
            # Assign reward from agent's (X=1) perspective
            if result == 'win':
                reward = 1.0 if game.current_player == 1 else -1.0
            else:
                reward = 0.0

            for step in trajectory:
                step[1] = reward
            break

    # Update running mean baseline (exponential moving average)
    alpha = 0.05
    mean_reward = (1 - alpha) * mean_reward + alpha * reward

    # REINFORCE update
    if trajectory:
        agent.mlp.zero_grad()
        loss = sum(
            -lp * (r - mean_reward)
            for lp, r in trajectory
            if r is not None
        )
        loss.backward()
        for p in agent.mlp.parameters():
            p.data -= lr * p.grad

    return reward, mean_reward


def play_episode_self_play(agent_x, agent_o, game, lr_x, lr_o, mean_x, mean_o):
    """
    One self-play episode: agent_x (X) vs agent_o (O).
    Returns (reward_x, reward_o, updated means).
    """
    game.reset()
    traj_x, traj_o = [], []

    while True:
        state = game.get_state()
        valid = game.get_valid_moves()

        if game.current_player == 1:
            action, log_prob = agent_x.select_action(state, valid, training=True)
            traj_x.append([log_prob, None])
        else:
            action, log_prob = agent_o.select_action(state, valid, training=True)
            traj_o.append([log_prob, None])

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
            loss = sum(-lp * (r - mean) for lp, r in traj if r is not None)
            loss.backward()
            for p in agent.mlp.parameters():
                p.data -= lr * p.grad

    update(agent_x, traj_x, mean_x, lr_x)
    update(agent_o, traj_o, mean_o, lr_o)

    return reward_x, reward_o, mean_x, mean_o


def eval_vs_random(agent, n_games=200):
    """Measure agent win/draw/loss rates against a random opponent."""
    game = TicTacToe()
    wins = draws = losses = 0
    for _ in range(n_games):
        game.reset()
        while True:
            state = game.get_state()
            valid = game.get_valid_moves()
            if game.current_player == 1:
                action, _ = agent.select_action(state, valid, training=False)
            else:
                action = random.choice(valid)
            result = game.make_move(action)
            if result != 'ongoing':
                if result == 'win':
                    if game.current_player == 1:
                        wins += 1
                    else:
                        losses += 1
                else:
                    draws += 1
                break
    return wins / n_games, draws / n_games, losses / n_games


def train(
    phase1_episodes=5000,
    phase2_episodes=10000,
    lr=0.01,
    eval_every=1000,
    save_path=AGENT_SAVE_PATH,
):
    print("=" * 60)
    print("tiny-brain TicTacToe — REINFORCE Training")
    print("=" * 60)
    print(f"Phase 1: {phase1_episodes} episodes vs random opponent")
    print(f"Phase 2: {phase2_episodes} episodes of self-play")
    print(f"Learning rate: {lr}")
    print()

    agent = NNAgent()
    game  = TicTacToe()
    mean_reward = 0.0
    win_history = []

    # ── Phase 1: vs random ────────────────────────────────────────────── #
    print("Phase 1: Training vs random opponent...")
    t0 = time.time()

    for ep in range(1, phase1_episodes + 1):
        reward, mean_reward = play_episode_vs_random(agent, game, lr, mean_reward)
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
    print("Phase 2: Self-play training...")
    # Start with a copy of the phase-1 agent as opponent
    agent_o = NNAgent()
    agent_o.mlp = agent.mlp   # share weights initially, they'll diverge
    import copy
    agent_o = copy.deepcopy(agent)

    mean_x = mean_o = 0.0
    t0 = time.time()

    for ep in range(1, phase2_episodes + 1):
        rx, ro, mean_x, mean_o = play_episode_self_play(
            agent, agent_o, game, lr, lr, mean_x, mean_o
        )

        # Periodically replace the weaker agent with the stronger one
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
