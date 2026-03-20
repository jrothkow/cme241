"""
plot_results_v5.py

Generate publication-style plots for the multi-capital life MDP:
  1) PPO vs exact-optimal reward distribution
  2) Time evolution of key states
  3) Policy behavior over time
  4) Terminal state distributions (PPO vs optimal)
  5) Interpretable policy visualization via shallow decision trees

Assumes your existing files are importable:
  - multi_capital_mdp_v6.py
  - gym_env_v5.py
  - evaluate_ppo_v5.py
and that your trained PPO model .zip is available.

Example usage
-------------
python plot_results_v5.py \
    --model ppo_nbhd2_v6 \
    --neighborhood 2 \
    --n-episodes 300 \
    --seed 42 \
    --outdir figures_v5

Optional: if you have already cached the exact solver outputs,
python final/plot_results.py \
    --model ppo_nbhd2_v6 \
    --solver-cache solver_outputs_nbhd2_v6.npz \
    --neighborhood 2 \
    --n-episodes 1000 \
    --seed 42 \
    --outdir figures_1

Notes
-----
- This script uses matplotlib only for plotting.
- For decision trees it uses sklearn. If needed:
    pip install scikit-learn
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from gym_env_v5 import LifeGymEnv5
from multi_capital_mdp_v6 import (
    StateV5,
    ActionV5,
    solve_mdp_fast_v5,
    policy_at_v5,
    load_solver_outputs_v5,
    save_solver_outputs_v5,
)
from evaluate_ppo_v5 import run_ppo_episodes, run_optimal_episodes


# ----------------------------
# Label helpers
# ----------------------------

MODE_LABELS = ["Recovery", "Strength", "Mixed/WC", "Endurance"]
INVEST_LABELS = ["0%", "10%", "20%"]
VOL_LABELS = ["Low", "Medium", "High"]
CONS_LABELS = ["Frugal", "Moderate", "Generous"]

STATE_FEATURE_NAMES = [
    "cash",
    "assets",
    "work_intensity",
    "energy",
    "strength",
    "work_cap",
    "endurance",
    "load",
    "injury",
    "time",
]


# ----------------------------
# Solver loading
# ----------------------------

def get_exact_solution(env: LifeGymEnv5, solver_cache: str | None):
    """
    Load cached exact solution if available, otherwise solve it.
    """
    if solver_cache is not None and Path(solver_cache).exists():
        V, PI, meta = load_solver_outputs_v5(solver_cache)
        print(f"Loaded cached exact solution from {solver_cache}")
        return V, PI

    print("Solving exact MDP via backward induction...")
    V, PI = solve_mdp_fast_v5(env.mdp)

    if solver_cache is not None:
        save_solver_outputs_v5(solver_cache, V, PI, env.mdp)
        print(f"Saved exact solution to {solver_cache}")

    return V, PI


# ----------------------------
# Rollout logging
# ----------------------------

def _state_to_vector(state: StateV5) -> np.ndarray:
    return np.array([
        state.cash,
        state.assets,
        state.work_intensity,
        state.energy,
        state.strength,
        state.work_cap,
        state.endurance,
        state.load,
        state.injury,
        state.time,
    ], dtype=np.int64)


def rollout_ppo_trajectory(
    model: PPO,
    env: LifeGymEnv5,
    seed: int = 0,
):
    """
    Log a single deterministic PPO rollout in the stochastic environment.
    """
    obs, _ = env.reset(seed=seed)
    done = False

    states = []
    actions = []
    rewards = []

    while not done:
        s = env._state
        a_raw, _ = model.predict(obs, deterministic=True)
        a = ActionV5(
            invest=int(a_raw[0]),
            mode=int(a_raw[1]),
            volume=int(a_raw[2]),
            consumption=int(a_raw[3]),
        )

        states.append(s)
        actions.append(a)

        obs, reward, terminated, truncated, _ = env.step(a_raw)
        rewards.append(float(reward))
        done = terminated or truncated

    terminal_state = env._state
    return states, actions, rewards, terminal_state


def rollout_optimal_trajectory(
    PI: np.ndarray,
    env: LifeGymEnv5,
    seed: int = 0,
):
    """
    Log a single rollout following the exact optimal policy in the same environment.
    """
    obs, _ = env.reset(seed=seed)
    done = False

    states = []
    actions = []
    rewards = []

    while not done:
        s = env._state
        a = policy_at_v5(PI, s)
        a_raw = np.array([a.invest, a.mode, a.volume, a.consumption], dtype=np.int64)

        states.append(s)
        actions.append(a)

        obs, reward, terminated, truncated, _ = env.step(a_raw)
        rewards.append(float(reward))
        done = terminated or truncated

    terminal_state = env._state
    return states, actions, rewards, terminal_state


def collect_policy_dataset(
    PI: np.ndarray,
    env: LifeGymEnv5,
    n_episodes: int = 500,
    seed: int = 0,
):
    """
    Collect (state, optimal action) supervised data by rolling out the exact policy.
    This gives an interpretable approximation of the policy via a shallow decision tree.
    """
    X = []
    y = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            s = env._state
            a = policy_at_v5(PI, s)

            X.append(_state_to_vector(s))
            y.append([a.invest, a.mode, a.volume, a.consumption])

            a_raw = np.array([a.invest, a.mode, a.volume, a.consumption], dtype=np.int64)
            obs, _, terminated, truncated, _ = env.step(a_raw)
            done = terminated or truncated

    return np.array(X, dtype=np.int64), np.array(y, dtype=np.int64)


# ----------------------------
# Plot 1: reward distribution
# ----------------------------

def plot_reward_distribution(
    ppo_rewards,
    opt_rewards,
    outpath: Path,
):
    plt.figure(figsize=(8, 5))
    bins = 25

    plt.hist(opt_rewards, bins=bins, alpha=0.55, density=False, label="Exact optimal")
    plt.hist(ppo_rewards, bins=bins, alpha=0.55, density=False, label="PPO")

    plt.axvline(np.mean(opt_rewards), linestyle="--", linewidth=2, label=f"Optimal mean = {np.mean(opt_rewards):.1f}")
    plt.axvline(np.mean(ppo_rewards), linestyle="--", linewidth=2, label=f"PPO mean = {np.mean(ppo_rewards):.1f}")

    plt.xlabel("Total episode reward")
    plt.ylabel("Count")
    plt.title("PPO vs exact-optimal reward distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()


# ----------------------------
# Plot 2: state trajectories
# ----------------------------

def _states_to_series(states: list[StateV5]):
    months = np.arange(1, len(states) + 1)
    return {
        "month": months,
        "cash": np.array([s.cash for s in states]),
        "assets": np.array([s.assets for s in states]),
        "energy": np.array([s.energy for s in states]),
        "strength": np.array([s.strength for s in states]),
        "work_cap": np.array([s.work_cap for s in states]),
        "endurance": np.array([s.endurance for s in states]),
        "load": np.array([s.load for s in states]),
        "injury": np.array([s.injury for s in states]),
        "work_intensity": np.array([s.work_intensity for s in states]),
    }


def plot_state_trajectories(
    ppo_states: list[StateV5],
    opt_states: list[StateV5],
    outpath: Path,
):
    ppo = _states_to_series(ppo_states)
    opt = _states_to_series(opt_states)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # Cash + assets
    ax = axes[0, 0]
    ax.plot(opt["month"], opt["cash"], linewidth=2, label="Optimal cash")
    ax.plot(opt["month"], opt["assets"], linewidth=2, label="Optimal assets")
    ax.plot(ppo["month"], ppo["cash"], linestyle="--", linewidth=2, label="PPO cash")
    ax.plot(ppo["month"], ppo["assets"], linestyle="--", linewidth=2, label="PPO assets")
    ax.set_title("Financial state")
    ax.set_ylabel("Bin")
    ax.legend(fontsize=8)

    # Energy + injury
    ax = axes[0, 1]
    ax.plot(opt["month"], opt["energy"], linewidth=2, label="Optimal energy")
    ax.plot(opt["month"], opt["injury"], linewidth=2, label="Optimal injury")
    ax.plot(ppo["month"], ppo["energy"], linestyle="--", linewidth=2, label="PPO energy")
    ax.plot(ppo["month"], ppo["injury"], linestyle="--", linewidth=2, label="PPO injury")
    ax.set_title("Recovery / health state")
    ax.set_ylabel("Level")
    ax.legend(fontsize=8)

    # Fitness
    ax = axes[1, 0]
    ax.plot(opt["month"], opt["strength"], linewidth=2, label="Optimal strength")
    ax.plot(opt["month"], opt["work_cap"], linewidth=2, label="Optimal work cap")
    ax.plot(opt["month"], opt["endurance"], linewidth=2, label="Optimal endurance")
    ax.plot(ppo["month"], ppo["strength"], linestyle="--", linewidth=2, label="PPO strength")
    ax.plot(ppo["month"], ppo["work_cap"], linestyle="--", linewidth=2, label="PPO work cap")
    ax.plot(ppo["month"], ppo["endurance"], linestyle="--", linewidth=2, label="PPO endurance")
    ax.set_title("Fitness state")
    ax.set_xlabel("Month")
    ax.set_ylabel("Level")
    ax.legend(fontsize=8, ncol=2)

    # Load + work intensity
    ax = axes[1, 1]
    ax.plot(opt["month"], opt["load"], linewidth=2, label="Optimal load")
    ax.plot(opt["month"], opt["work_intensity"], linewidth=2, label="Optimal work intensity")
    ax.plot(ppo["month"], ppo["load"], linestyle="--", linewidth=2, label="PPO load")
    ax.plot(ppo["month"], ppo["work_intensity"], linestyle="--", linewidth=2, label="PPO work intensity")
    ax.set_title("Stress state")
    ax.set_xlabel("Month")
    ax.set_ylabel("Level")
    ax.legend(fontsize=8)

    fig.suptitle("Time evolution of key states", fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Plot 3: policy behavior
# ----------------------------

def _actions_to_series(actions: list[ActionV5]):
    months = np.arange(1, len(actions) + 1)
    return {
        "month": months,
        "invest": np.array([a.invest for a in actions]),
        "mode": np.array([a.mode for a in actions]),
        "volume": np.array([a.volume for a in actions]),
        "consumption": np.array([a.consumption for a in actions]),
    }


def plot_policy_behavior(
    ppo_actions: list[ActionV5],
    opt_actions: list[ActionV5],
    outpath: Path,
):
    ppo = _actions_to_series(ppo_actions)
    opt = _actions_to_series(opt_actions)

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)

    specs = [
        ("invest", "Investment action"),
        ("mode", "Training mode"),
        ("volume", "Training volume"),
        ("consumption", "Consumption action"),
    ]

    for ax, (key, title) in zip(axes, specs):
        ax.step(opt["month"], opt[key], where="mid", linewidth=2, label="Optimal")
        ax.step(ppo["month"], ppo[key], where="mid", linewidth=2, linestyle="--", label="PPO")
        ax.set_ylabel(title)
        ax.set_yticks(sorted(set(np.r_[opt[key], ppo[key]])))
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Month")
    fig.suptitle("Policy behavior over time", fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Plot 4: terminal distributions
# ----------------------------

def _plot_discrete_overlay(ax, ppo_vals, opt_vals, n_levels, title):
    bins = np.arange(-0.5, n_levels + 0.5, 1.0)
    ax.hist(opt_vals, bins=bins, alpha=0.55, density=True, label="Exact optimal")
    ax.hist(ppo_vals, bins=bins, alpha=0.55, density=True, label="PPO")
    ax.set_xticks(range(n_levels))
    ax.set_title(title)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)


def plot_terminal_distributions(
    ppo_terminal: dict[str, list[int]],
    opt_terminal: dict[str, list[int]],
    outpath: Path,
):
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))

    specs = [
        ("cash", 21, "Final cash"),
        ("assets", 31, "Final assets"),
        ("strength", 4, "Final strength"),
        ("work_cap", 4, "Final work capacity"),
        ("endurance", 4, "Final endurance"),
        ("injury", 3, "Final injury"),
    ]

    for ax, (key, n_levels, title) in zip(axes.flat, specs):
        _plot_discrete_overlay(ax, ppo_terminal[key], opt_terminal[key], n_levels, title)

    fig.suptitle("Terminal state distributions: PPO vs exact optimal", fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Plot 5: decision trees
# ----------------------------

def plot_policy_decision_trees(
    X: np.ndarray,
    y: np.ndarray,
    outpath: Path,
    max_depth: int = 3,
    min_samples_leaf: int = 50,
):
    """
    Fit one shallow decision tree per action dimension.
    This gives a compact, human-readable approximation of the exact policy.
    """
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    titles = [
        "Investment rule",
        "Training-mode rule",
        "Training-volume rule",
        "Consumption rule",
    ]
    class_names = [
        INVEST_LABELS,
        MODE_LABELS,
        VOL_LABELS,
        CONS_LABELS,
    ]

    fig, axes = plt.subplots(2, 2, figsize=(24, 14))
    axes = axes.ravel()

    for j in range(4):
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=0,
        )
        clf.fit(X, y[:, j])

        plot_tree(
            clf,
            ax=axes[j],
            feature_names=STATE_FEATURE_NAMES,
            class_names=class_names[j],
            filled=True,
            rounded=True,
            impurity=False,
            fontsize=9,
        )
        acc = clf.score(X, y[:, j])
        axes[j].set_title(f"{titles[j]}  (train acc = {acc:.3f})")

    fig.suptitle("Interpretable decision-tree approximation of the exact policy", fontsize=16)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to PPO model zip or basename")
    parser.add_argument("--neighborhood", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--n-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="figures_v5")
    parser.add_argument("--solver-cache", type=str, default=None,
                        help="Optional .npz cache for exact solver outputs")
    parser.add_argument("--tree-episodes", type=int, default=500,
                        help="Number of exact-policy episodes used to fit decision trees")
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--tree-min-leaf", type=int, default=50)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Separate envs keep episode generation clean and reproducible
    env_for_solver = LifeGymEnv5(neighborhood=args.neighborhood)
    env_ppo = LifeGymEnv5(neighborhood=args.neighborhood)
    env_opt = LifeGymEnv5(neighborhood=args.neighborhood)
    env_traj_ppo = LifeGymEnv5(neighborhood=args.neighborhood)
    env_traj_opt = LifeGymEnv5(neighborhood=args.neighborhood)
    env_tree = LifeGymEnv5(neighborhood=args.neighborhood)

    model = PPO.load(args.model)

    # Exact solution
    V, PI = get_exact_solution(env_for_solver, args.solver_cache)

    # ---------------------------------------------------------
    # 1) Reward distributions
    # ---------------------------------------------------------
    print("Running PPO evaluation episodes...")
    (
        ppo_rewards,
        ppo_cash,
        ppo_assets,
        ppo_strength,
        ppo_work_cap,
        ppo_endurance,
        ppo_load,
        ppo_injury,
    ) = run_ppo_episodes(model, env_ppo, n_episodes=args.n_episodes, seed=args.seed)

    print("Running exact-optimal evaluation episodes...")
    (
        opt_rewards,
        opt_cash,
        opt_assets,
        opt_strength,
        opt_work_cap,
        opt_endurance,
        opt_load,
        opt_injury,
    ) = run_optimal_episodes(PI, env_opt, n_episodes=args.n_episodes, seed=args.seed)

    plot_reward_distribution(
        ppo_rewards=ppo_rewards,
        opt_rewards=opt_rewards,
        outpath=outdir / "reward_distribution.png",
    )

    # ---------------------------------------------------------
    # 2) Key-state trajectory subplots
    # 3) Policy behavior over time
    # ---------------------------------------------------------
    # Use same seed so both rollouts start from the same initial state distribution
    # and experience the same RNG seeding scheme at reset.
    ppo_states, ppo_actions, ppo_step_rewards, ppo_terminal_state = rollout_ppo_trajectory(
        model, env_traj_ppo, seed=args.seed
    )
    opt_states, opt_actions, opt_step_rewards, opt_terminal_state = rollout_optimal_trajectory(
        PI, env_traj_opt, seed=args.seed
    )

    plot_state_trajectories(
        ppo_states=ppo_states,
        opt_states=opt_states,
        outpath=outdir / "state_trajectories.png",
    )

    plot_policy_behavior(
        ppo_actions=ppo_actions,
        opt_actions=opt_actions,
        outpath=outdir / "policy_behavior_over_time.png",
    )

    # ---------------------------------------------------------
    # 4) Terminal distributions
    # ---------------------------------------------------------
    ppo_terminal = {
        "cash": ppo_cash,
        "assets": ppo_assets,
        "strength": ppo_strength,
        "work_cap": ppo_work_cap,
        "endurance": ppo_endurance,
        "load": ppo_load,
        "injury": ppo_injury,
    }
    opt_terminal = {
        "cash": opt_cash,
        "assets": opt_assets,
        "strength": opt_strength,
        "work_cap": opt_work_cap,
        "endurance": opt_endurance,
        "load": opt_load,
        "injury": opt_injury,
    }

    plot_terminal_distributions(
        ppo_terminal=ppo_terminal,
        opt_terminal=opt_terminal,
        outpath=outdir / "terminal_distributions.png",
    )

    # ---------------------------------------------------------
    # 5) Decision-tree approximation of exact policy
    # ---------------------------------------------------------
    print("Collecting supervised dataset for decision-tree policy visualization...")
    X, y = collect_policy_dataset(
        PI=PI,
        env=env_tree,
        n_episodes=args.tree_episodes,
        seed=args.seed,
    )

    plot_policy_decision_trees(
        X=X,
        y=y,
        outpath=outdir / "policy_decision_trees.png",
        max_depth=args.tree_depth,
        min_samples_leaf=args.tree_min_leaf,
    )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------
    print("\nSaved figures:")
    for fp in [
        outdir / "reward_distribution.png",
        outdir / "state_trajectories.png",
        outdir / "policy_behavior_over_time.png",
        outdir / "terminal_distributions.png",
        outdir / "policy_decision_trees.png",
    ]:
        print(f"  - {fp}")

    print("\nQuick summary:")
    print(f"  PPO mean reward     : {np.mean(ppo_rewards):.2f} ± {np.std(ppo_rewards):.2f}")
    print(f"  Optimal mean reward : {np.mean(opt_rewards):.2f} ± {np.std(opt_rewards):.2f}")
    print(f"  Reward gap          : {np.mean(opt_rewards) - np.mean(ppo_rewards):.2f}")


if __name__ == "__main__":
    main()