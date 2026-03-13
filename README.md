# tiny-brain

A from-scratch neural network library — pure Python, zero dependencies.
Inspired by Andrej Karpathy's [micrograd lecture](https://youtu.be/VMj-3S1tku0).

Includes a TicTacToe game with a **live neural network visualizer** in the browser.

![tiny-brain UI](assets/screenshot.png)

*Left: TicTacToe board with purple heatmap showing where the AI wants to play. Right: live neural network diagram — input layer (board cells), two hidden layers (neurons light up by activation strength), output layer (move probabilities).*

---

## What's inside

```
tinybrain/          Core library (pure Python, no deps)
  engine.py         Value class — scalar autograd engine
  nn.py             Neuron, Layer, MLP

demo.py             Karpathy-style validation (run this first)

games/tictactoe/
  game.py           Board logic
  agent.py          NNAgent wrapping MLP
  train.py          REINFORCE training loop
  server.py         Flask web app (game + brain visualizer)
  templates/
    index.html      Browser UI with live NN visualization
```

---

## Quickstart

```bash
# 1. Install Flask (only external dependency)
pip install flask

# 2. Validate the core library
python demo.py

# 3. Train the TicTacToe agent (~15 min for full training, or try quick mode)
python games/tictactoe/train.py

# 4. Play against it in the browser
python games/tictactoe/server.py
# → open http://localhost:5000
```

---

## How it works

### The autograd engine (`engine.py`)

Every number is wrapped in a `Value` object. When you do math with `Value`s,
a computation graph is built silently in the background. Calling `.backward()`
on the final result walks the graph in reverse, computing how much each
input contributed to the output — that's the gradient.

```python
from tinybrain import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b + Value(10.0)   # builds a graph
c.backward()

print(a.grad)   # dc/da = b = -3.0
print(b.grad)   # dc/db = a =  2.0
```

### The MLP (`nn.py`)

```python
from tinybrain import MLP

model = MLP(2, [16, 16, 1])   # 2 inputs → 2 hidden layers → 1 output
out = model([1.0, -1.0])      # forward pass, returns a Value
out.backward()                # compute all gradients
model.zero_grad()             # reset before next step
```

### Training TicTacToe

The agent uses **REINFORCE** (policy gradient):
1. Play a full game, recording `(log_prob_of_action, reward)` for each move
2. Win = +1, Draw = 0, Lose = -1
3. `loss = -sum(log_prob * advantage)` where `advantage = reward - running_mean`
4. `loss.backward()` flows gradients through the entire network
5. SGD nudges weights to make winning moves more likely

---

## The brain visualizer

When you play in the browser, you see the neural network thinking in real time:

- **Input layer** (9 neurons) — board cells: purple=X, red=O, dark=empty
- **Hidden layers** (32 neurons each) — brightness = how strongly each neuron fires
- **Output layer** (9 neurons) — how much the AI wants each cell (probability)
- **Board heatmap** — purple overlay on cells the AI prefers most

---

## Adding new games

The `tinybrain/` library is game-agnostic. To add a new game:
1. Create `games/<your_game>/`
2. Import `MLP` from `tinybrain`
3. Build an agent, a training loop, and a Flask server
