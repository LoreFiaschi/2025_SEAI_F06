# MiniChessMCTS

A **minimal yet complete** reference implementation of Monte‑Carlo Tree Search (MCTS) combined with a lightweight convolutional **value network** trained via self‑play reinforcement learning for *Gardner 5×5 Mini‑Chess*.

## Highlights

* Pure‑Python rules & move generator – no external chess libraries.
* PUCT‑style MCTS with batched neural evaluations (`mcts_pt.py`).
* End‑to‑end self‑play RL pipeline in < 400 lines (`train.py`).
* TrueSkill arena to keep only stronger checkpoints.
* Play either against the default heuristic engine **or against any saved model checkpoint**.
* Tiny CNN (\~6 M params) that runs comfortably on CPU or GPU.

## Quick Start

| Task                              | Command                                                                 |
| --------------------------------- | ----------------------------------------------------------------------- |
| Play against the default engine   | `python ChessGame.py`                                                   |
| Play against a **trained model**  | `python demo.py --ckpt checkpoints/best_50.pth`                         |
| Watch AI vs AI (two models)       | `python demo_dual.py --ckpt_white best_10.pth --ckpt_black best_20.pth` |
| Train from scratch (self‑play RL) | `python train.py --episodes 10000 --checkpoint_dir checkpoints/`        |

## Reinforcement Learning Pipeline

1. **Self‑play** – The current network plays both sides via MCTS‑PUCT, storing `(state, value)` pairs where *value* ∈ {‑1, 0, +1} is the final result from the perspective of the player to move.
2. **Training** – After *N* games the value network is updated with mean‑squared error on the collected targets.
3. **Evaluation** – The freshly trained model faces the previous best in a TrueSkill arena; if its rating is higher it becomes the new champion and its weights are checkpointed.
4. The loop repeats until the episode or time budget is exhausted.

All of the above logic lives in `train.py`, with hyper‑parameters consolidated in `config.py`.

## Citation

If you use this code, please cite:

> Comini & Vittori, *A Study of MCTS for 5×5 Mini‑Chess*, SEAI‑NS‑RL 2025.
