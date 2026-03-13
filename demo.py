"""
demo.py — Karpathy-style validation of the tiny-brain library.

Run with:  python demo.py

Three parts:
  A. Manual expression graph  (matches Karpathy's first example exactly)
  B. Single neuron walkthrough (matches Karpathy's neuron demo)
  C. MLP trains on XOR        (proves the full training loop works)

No external dependencies — pure Python.
"""

from tinybrain.engine import Value
from tinybrain.nn import MLP


# ═══════════════════════════════════════════════════════════════════════════ #
#  PART A — Manual expression graph                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def part_a():
    print("=" * 60)
    print("PART A: Manual expression graph")
    print("=" * 60)
    print()
    print("Building:  L = (a*b + c) * f")
    print("         = (2*-3 + 10) * -2")
    print("         = (  -6 + 10) * -2")
    print("         = 4 * -2 = -8")
    print()

    a = Value(2.0,  label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    f = Value(-2.0, label='f')

    e = a * b;      e.label = 'e'   # e = -6
    d = e + c;      d.label = 'd'   # d = 4
    L = d * f;      L.label = 'L'   # L = -8

    print(f"L.data = {L.data}")     # should be -8.0
    print()

    L.backward()

    print("Gradients (computed by backprop):")
    print(f"  dL/dL = {L.grad:.1f}  (always 1 — the root)")
    print(f"  dL/dd = {d.grad:.1f}  (= f = -2)")
    print(f"  dL/df = {f.grad:.1f}  (= d = 4)")
    print(f"  dL/de = {e.grad:.1f}  (= dL/dd * dd/de = -2 * 1 = -2)")
    print(f"  dL/dc = {c.grad:.1f}  (= dL/dd * dd/dc = -2 * 1 = -2)")
    print(f"  dL/da = {a.grad:.1f}  (= dL/de * de/da = -2 * b = -2 * -3 = 6)")
    print(f"  dL/db = {b.grad:.1f}  (= dL/de * de/db = -2 * a = -2 *  2 = -4)")
    print()

    # Assertions against analytic results
    assert abs(L.data - (-8.0)) < 1e-9,  f"L.data wrong: {L.data}"
    assert abs(a.grad -   6.0)  < 1e-9,  f"a.grad wrong: {a.grad}"
    assert abs(b.grad - (-4.0)) < 1e-9,  f"b.grad wrong: {b.grad}"
    assert abs(c.grad - (-2.0)) < 1e-9,  f"c.grad wrong: {c.grad}"
    assert abs(f.grad -   4.0)  < 1e-9,  f"f.grad wrong: {f.grad}"

    print("✓ All Part A assertions passed.\n")


# ═══════════════════════════════════════════════════════════════════════════ #
#  PART B — Single neuron with tanh  (Karpathy's exact demo)                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def part_b():
    print("=" * 60)
    print("PART B: Single neuron (matches Karpathy's values)")
    print("=" * 60)
    print()
    print("One neuron: o = tanh(x1*w1 + x2*w2 + b)")
    print("Inputs:  x1=2.0, x2=0.0")
    print("Weights: w1=-3.0, w2=1.0")
    print("Bias:    b=6.8813735870195432")
    print()

    x1 = Value(2.0,  label='x1')
    x2 = Value(0.0,  label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0,  label='w2')
    b  = Value(6.8813735870195432, label='b')

    x1w1     = x1 * w1;           x1w1.label = 'x1*w1'
    x2w2     = x2 * w2;           x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2;      x1w1x2w2.label = 'x1w1 + x2w2'
    n        = x1w1x2w2 + b;     n.label = 'n'
    o        = n.tanh();          o.label = 'o'

    print(f"n.data = {n.data:.4f}  (pre-tanh sum)")
    print(f"o.data = {o.data:.4f}  (tanh output, should be ~0.7071)")
    print()

    o.backward()

    print("Gradients:")
    print(f"  o.grad  = {o.grad:.4f}  (1.0 — root)")
    print(f"  n.grad  = {n.grad:.4f}  (1 - tanh² ≈ 0.5)")
    print(f"  b.grad  = {b.grad:.4f}  (same as n.grad)")
    print(f"  w1.grad = {w1.grad:.4f}  (n.grad * x1 ≈ 0.5 * 2 = 1.0)")
    print(f"  x1.grad = {x1.grad:.4f}  (n.grad * w1 ≈ 0.5 * -3 = -1.5)")
    print(f"  w2.grad = {w2.grad:.4f}  (n.grad * x2 = 0.5 * 0 = 0.0)")
    print(f"  x2.grad = {x2.grad:.4f}  (n.grad * w2 = 0.5 * 1 = 0.5)")
    print()

    # Karpathy's known values from the lecture
    assert abs(o.data  - 0.7071) < 1e-4,  f"o.data wrong: {o.data}"
    assert abs(w1.grad - 1.0)    < 1e-4,  f"w1.grad wrong: {w1.grad}"
    assert abs(x1.grad - (-1.5)) < 1e-4,  f"x1.grad wrong: {x1.grad}"
    assert abs(w2.grad - 0.0)    < 1e-4,  f"w2.grad wrong: {w2.grad}"

    print("✓ All Part B assertions passed.\n")


# ═══════════════════════════════════════════════════════════════════════════ #
#  PART C — MLP trains on XOR  (4 data points, pure Python)                  #
# ═══════════════════════════════════════════════════════════════════════════ #

def part_c():
    print("=" * 60)
    print("PART C: MLP learns XOR")
    print("=" * 60)
    print()
    print("XOR is the classic test because a linear model CAN'T solve it —")
    print("you need at least one hidden layer to carve the non-linear boundary.")
    print()
    print("Data:  [0,0]→-1  [0,1]→+1  [1,0]→+1  [1,1]→-1")
    print("Loss:  hinge = max(0, 1 - y * pred)  (no log needed)")
    print()

    # Dataset: XOR truth table, labels in {-1, +1} (hinge loss convention)
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [-1,      1,      1,     -1]

    # Small MLP: 2 inputs → 8 hidden → 8 hidden → 1 output
    model = MLP(2, [8, 8, 1])
    n_params = len(model.parameters())
    print(f"Model: MLP(2, [8, 8, 1])  —  {n_params} parameters\n")

    lr = 0.05
    n_epochs = 300

    for epoch in range(n_epochs):
        # Forward pass
        ypred = [model(x) for x in X]

        # Hinge loss: penalise predictions that are wrong or low-confidence.
        # max(0, 1 - y*pred) = 0 when correctly classified with margin ≥ 1.
        data_loss = sum((Value(1.0) - yi * yp).relu()
                        for yi, yp in zip(y, ypred)) * (1.0 / len(y))

        # L2 regularisation: gently pull weights toward zero to prevent
        # the network from relying too heavily on any single weight.
        reg_loss = 1e-4 * sum(p * p for p in model.parameters())

        loss = data_loss + reg_loss

        # Backward pass
        model.zero_grad()   # MUST zero before backward or grads accumulate
        loss.backward()

        # SGD parameter update:  move each weight a small step against its gradient
        for p in model.parameters():
            p.data -= lr * p.grad

        if epoch % 30 == 0 or epoch == n_epochs - 1:
            acc = sum(
                (yi > 0) == (yp.data > 0)
                for yi, yp in zip(y, ypred)
            ) / len(y)
            preds_str = "  ".join(f"{yp.data:+.3f}" for yp in ypred)
            print(f"  epoch {epoch:3d}: loss={loss.data:.4f}  acc={acc:.0%}  "
                  f"preds=[{preds_str}]")

    print()
    final_acc = sum(
        (yi > 0) == (yp.data > 0)
        for yi, yp in zip(y, [model(x) for x in X])
    ) / len(y)

    print(f"Final accuracy: {final_acc:.0%}")
    assert final_acc == 1.0, \
        f"Expected 100% on XOR after {n_epochs} epochs, got {final_acc:.0%}"
    print("✓ All Part C assertions passed.\n")


# ═══════════════════════════════════════════════════════════════════════════ #
#  Run all parts                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

if __name__ == '__main__':
    print()
    print("  tiny-brain demo  —  inspired by Karpathy's micrograd lecture")
    print("  https://youtu.be/VMj-3S1tku0")
    print()

    part_a()
    part_b()
    part_c()

    print("=" * 60)
    print("All demos passed. tiny-brain is working correctly.")
    print("=" * 60)
