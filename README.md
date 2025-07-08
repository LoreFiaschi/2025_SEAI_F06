# 🧠 MiniChessMCTS

A **minimal yet complete** implementation of Monte Carlo Tree Search (MCTS) combined with a lightweight convolutional **value network** trained through self-play reinforcement learning — built for *Gardner 5×5 Mini‑Chess*.

---

## 🚀 Features

* ♟ **Pure Python** rule engine & move generator — no external chess libraries required.
* 🔍 **PUCT-based MCTS** with batched neural evaluations (`mcts_pt.py`).
* 🧠 **Self-play RL pipeline** in under 400 lines (`train.py`).
* 🏆 **TrueSkill-based arena**: only stronger models survive.
* 🎮 **Human vs AI** or **AI vs AI** play supported.
* ⚡ **Tiny CNN (\~6M params)** runs smoothly on CPU/GPU.

---

## ⚡ Quick Start

| Task                           | Command                                                                 |
| ------------------------------ | ----------------------------------------------------------------------- |
| Play against default engine    | `python ChessGame.py`                                                   |
| Play vs trained model          | `python demo.py --ckpt checkpoints/best_50.pth`                         |
| Watch AI vs AI (2 models)      | `python demo_dual.py --ckpt_white best_10.pth --ckpt_black best_20.pth` |
| Train from scratch (self-play) | `python train.py --episodes 10000 --checkpoint_dir checkpoints/`        |

---

## 🧩 RL Pipeline Overview

1. **Self-Play**
   The current network plays both sides using MCTS‑PUCT, collecting `(state, value)` pairs. The *value* ∈ {‑1, 0, +1} reflects the game outcome for the current player.

2. **Training**
   After *N* games, the value network is updated via mean squared error using the collected targets.

3. **Evaluation**
   The newly trained network competes against the current best using a TrueSkill arena. If it wins, it becomes the new best and is saved.

4. **Repeat**
   The loop continues until the episode/time budget is exhausted.

> Core logic in `train.py`. Hyperparameters in `config.py`.

---

## 📄 Citation

If you use this project in academic or research settings, please cite:

> **Comini & Vittori**
> *A Study of MCTS for 5×5 Mini‑Chess*, SEAI‑NS‑RL 2025.

