"""
games/nurikabe/train_pytorch.py

Train a Nurikabe iterative solver using PyTorch (MPS on Apple Silicon).

Architecture: 50 → 256 → 256 → 25
  Input  : 50 floats = 25 clue channel + 25 state channel
             clue[i]  = size/5.0  for clue cells, 0 elsewhere
             state[i] = +1 (white locked), -1 (black locked), 0 (unknown)
  Hidden : two ReLU layers of 256 neurons
  Output : 25 logits → sigmoid → P(cell is white/island)

Training:
  - Supervised: BCEWithLogitsLoss against known solutions
  - For each puzzle, randomly reveal 0–REVEAL_MAX already-solved cells
    → NN learns to use partial information (iterative inference)
  - Adam + CosineAnnealingLR, 150 epochs, 50k puzzles

Accuracy metric:
  Full-puzzle accuracy: run the complete iterative solve on each val puzzle,
  compare final solution to ground truth. ALL 49 cells must match.

After training, weights are exported in tiny-brain MLP.parameters() order
so NurikabeAgent can load them for inference.
"""

import os
import sys
import time
import pickle
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from games.nurikabe.puzzle import generate_puzzle, SIZE


# ── Config ────────────────────────────────────────────────────────────────── #

LAYER_SIZES  = [50, 256, 256, 25]   # 5×5: 25 clue + 25 state → 25 outputs
N_PUZZLES    = 100_000              # more puzzles since 5×5 generates fast
BATCH_SIZE   = 256
EPOCHS       = 150
LR           = 1e-3
WEIGHT_DECAY = 1e-4
REVEAL_MAX   = 20                   # 5×5 has 25 cells, reveal up to 20
SAVE_PATH    = os.path.join(os.path.dirname(__file__), 'agent.pkl')


# ── Device ────────────────────────────────────────────────────────────────── #

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Input builder ─────────────────────────────────────────────────────────── #

def build_input(clues, state_dict, size=SIZE):
    """
    Build a 98-float input vector from clues + partial state.

    clues      : dict {cell_index: island_size}
    state_dict : dict {cell_index: +1 (white) or -1 (black)}
                 (unknown cells absent or mapped to 0)

    Returns list of 98 floats:
        [clue_channel (49)] + [state_channel (49)]
    """
    clue_ch  = [0.0] * size
    state_ch = [0.0] * size
    for ci, sz in clues.items():
        clue_ch[ci] = sz / 5.0
    for ci, val in state_dict.items():
        state_ch[ci] = float(val)   # +1.0 or -1.0
    return clue_ch + state_ch       # 98 floats


# ── Dataset ───────────────────────────────────────────────────────────────── #

class NurikabeDataset(Dataset):
    """
    Pre-generates `n_puzzles` training examples.

    Each example has a random number of already-solved cells revealed (0–REVEAL_MAX).
    This teaches the NN to predict the *remaining* cells given partial information.
    """

    def __init__(self, n_puzzles, seed=42):
        random.seed(seed)
        self.inputs  = []
        self.targets = []

        print(f"  Generating {n_puzzles:,} puzzles with random partial reveals…",
              flush=True)
        t0 = time.time()

        for i in range(n_puzzles):
            clues, solution = generate_puzzle()

            # Clue channel
            clue_ch = [0.0] * SIZE
            for ci, sz in clues.items():
                clue_ch[ci] = sz / 5.0

            # Randomly reveal K non-clue cells (correct values from solution)
            non_clue = [c for c in range(SIZE) if c not in clues]
            k = random.randint(0, min(REVEAL_MAX, len(non_clue)))
            revealed = random.sample(non_clue, k)

            state_ch = [0.0] * SIZE
            for ci in revealed:
                state_ch[ci] = 1.0 if solution[ci] == 1 else -1.0

            inp = clue_ch + state_ch      # 98 floats
            self.inputs.append(inp)
            self.targets.append([float(v) for v in solution])

            if (i + 1) % 10_000 == 0:
                print(f"    {i + 1:,} puzzles ({time.time() - t0:.1f}s)",
                      flush=True)

        self.inputs  = torch.tensor(self.inputs,  dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        print(f"  Dataset ready: {len(self.inputs):,} puzzles in "
              f"{time.time() - t0:.1f}s", flush=True)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]


# ── Model ─────────────────────────────────────────────────────────────────── #

class NurikabeNet(nn.Module):
    """
    50 → 256 → 256 → 25 MLP.
    ReLU hidden activations; raw logits out (BCEWithLogitsLoss handles sigmoid).
    Input: [clue_channel (25)] + [state_channel (25)]
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(50, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 25),
        )

    def forward(self, x):
        return self.net(x)


# ── Iterative solver (for validation accuracy) ────────────────────────────── #

def iterative_solve_torch(model, clues, device):
    """
    Run the full iterative solve on one puzzle using the PyTorch model.
    Used during validation to compute full-puzzle accuracy.

    Returns list of 49 ints (0/1).
    """
    model.eval()
    state = {}
    # Clue cells are always white
    for ci in clues:
        state[ci] = 1

    unknown = [c for c in range(SIZE) if c not in clues]

    with torch.no_grad():
        while unknown:
            inp = build_input(clues, state)
            x   = torch.tensor([inp], dtype=torch.float32).to(device)
            logits = model(x)[0]
            probs  = torch.sigmoid(logits).cpu().tolist()

            # Pick most confident unknown cell
            best_cell = max(unknown, key=lambda c: abs(probs[c] - 0.5))
            state[best_cell] = 1 if probs[best_cell] >= 0.5 else 0
            unknown.remove(best_cell)

    return [state.get(i, 0) for i in range(SIZE)]


# ── Weight export ─────────────────────────────────────────────────────────── #

def export_weights(model):
    """
    Flatten weights in tiny-brain MLP.parameters() order:
      for each Linear layer → for each neuron i → [weights..., bias]
    """
    flat = []
    linear_layers = [m for m in model.net.modules() if isinstance(m, nn.Linear)]
    for linear in linear_layers:
        w = linear.weight.detach().cpu()
        b = linear.bias.detach().cpu()
        for i in range(w.shape[0]):
            flat.extend(w[i].tolist())
            flat.append(b[i].item())
    return flat


def save_agent(model, path):
    weights = export_weights(model)
    data = {'layer_sizes': LAYER_SIZES, 'weights': weights}
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved agent → {path}  ({len(weights):,} weights)", flush=True)


# ── Training loop ─────────────────────────────────────────────────────────── #

def train():
    print("=" * 62)
    print("tiny-brain Nurikabe — Iterative NN Training")
    print("=" * 62)

    device = get_device()
    print(f"Device     : {device}", flush=True)
    print(f"Architecture: {' → '.join(str(s) for s in LAYER_SIZES)}", flush=True)
    print(f"Puzzles    : {N_PUZZLES:,}  |  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}",
          flush=True)
    print(f"Reveal     : 0–{REVEAL_MAX} cells per example", flush=True)
    print()

    dataset = NurikabeDataset(N_PUZZLES)

    n_val   = len(dataset) // 10
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=512,        shuffle=False)

    model     = NurikabeNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}", flush=True)
    print()

    # Extract val puzzles for iterative accuracy (sample 200 to keep it fast)
    val_indices = list(range(n_val))
    random.shuffle(val_indices)
    acc_sample  = min(200, n_val)
    val_clues_solutions = []
    raw_val_ds = val_ds.dataset
    for idx in val_indices[:acc_sample]:
        # Recover clues from the input vector (clue_ch = first 49 floats)
        inp_vec  = raw_val_ds.inputs[val_ds.indices[idx]].tolist()
        sol_vec  = raw_val_ds.targets[val_ds.indices[idx]].tolist()
        clue_ch  = inp_vec[:SIZE]
        clues    = {i: round(v * 5) for i, v in enumerate(clue_ch) if v > 0.01}
        solution = [int(v) for v in sol_vec]
        val_clues_solutions.append((clues, solution))

    best_val_loss = float('inf')
    t0 = time.time()

    print(f"  {'Epoch':>5}  {'train':>8}  {'val':>8}  {'puzzle_acc':>10}  {'time':>6}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*6}")

    for epoch in range(1, EPOCHS + 1):
        # ── train ── #
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train

        # ── val loss ── #
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * len(x)
        val_loss /= n_val

        # ── full-puzzle accuracy (iterative solve on sample) ── #
        correct = 0
        for clues, solution in val_clues_solutions:
            pred = iterative_solve_torch(model, clues, device)
            if pred == solution:
                correct += 1
        puzzle_acc = correct / acc_sample

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_agent(model, SAVE_PATH)

        elapsed = time.time() - t0
        print(f"  {epoch:5d}  {train_loss:8.4f}  {val_loss:8.4f}  "
              f"{puzzle_acc:10.1%}  {elapsed:5.0f}s", flush=True)

    print()
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Run:  python3 games/nurikabe/server.py  to play.")


if __name__ == '__main__':
    train()
