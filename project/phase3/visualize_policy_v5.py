"""
Policy visualization for the v5 Multi-Capital Life MDP.

Supports the exact backward-induction policy, a trained PPO model, or both.

Usage
-----
  # Exact optimal only (default)
  python -m v5.visualize_policy_v5 --cash 8 --assets 5 --energy 2 --strength 2 --work-cap 2 --endurance 2

  # PPO policy
  python -m v5.visualize_policy_v5 --cash 0 --assets 0 --energy 2 --strength 2 --work-cap 2 --endurance 2 \\
      --policy ppo --model v5/ppo_nbhd2_v5

  # Side-by-side (optimal then PPO)
  python -m v5.visualize_policy_v5 --cash 0 --assets 0 --energy 2 --strength 2 --work-cap 2 --endurance 2 \\
      --policy both --model v5/ppo_nbhd2_v5

  # Stochastic rollout
  python -m v5.visualize_policy_v5 --cash 8 --assets 5 --energy 2 --strength 2 --work-cap 2 --endurance 2 \\
      --sample --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Add phase3/ to path so v5.* imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_capital_mdp_v6 import (
    MultiCapitalMDPv5,
    StateV5,
    ActionV5,
    solve_mdp_fast_v5,
    policy_at_v5,
    value_at_v5,
    save_solver_outputs_v5
)


# ============================================================================
# Label helpers
# ============================================================================

CASH_UNIT  = 2_500   # one cash bin  = $10k
ASSET_UNIT = 5_000   # one asset bin = $20k

IW_LABELS    = ["Low", "Mod", "High"]
LEVEL_LABELS = ["Low", "Med", "High"]          # energy / volume / consumption
FIT_LABELS   = ["0", "1", "2", "3"]           # strength / work_cap / endurance
LOAD_LABELS  = ["0", "1", "2", "3", "4"]      # load
INJ_LABELS   = ["✓", "○", "✗"]               # injury: healthy / minor / major
MODE_LABELS  = ["Rec", "Str", "Mix", "End"]
INVEST_LABELS = ["0%", "10%", "20%"]
NBHD_LABELS   = ["Budget", "Mid-tier", "Premium"]


def cash_dollars(c: int) -> str:
    return f"${c * CASH_UNIT // 1_000:,}k"


def asset_dollars(a: int) -> str:
    return f"${a * ASSET_UNIT // 1_000:,}k"


def level_bar4(v: int) -> str:
    """Four-cell bar for 0–3 levels."""
    return "".join("█" if i <= v else "░" for i in range(4))


def level_bar5(v: int) -> str:
    """Five-cell bar for 0–4 levels (load)."""
    return "".join("█" if i <= v else "░" for i in range(5))


# ============================================================================
# Rollout functions
# ============================================================================

def _rollout_core(
    mdp: MultiCapitalMDPv5,
    V: np.ndarray,
    PI: np.ndarray,
    start: StateV5,
    rng: np.random.Generator | None,
    ppo_model=None,
    ppo_obs_fn=None,
) -> list[dict]:
    """
    Shared rollout engine.

    - If ppo_model is provided: actions come from PPO (deterministic greedy).
    - Otherwise: actions come from the exact policy PI.
    - If rng is None: use argmax for next-state sampling (expected path).
    - If rng is provided: sample from transition distributions.
    """
    rows = []
    state = start

    for t in range(mdp.time_horizon):
        state = StateV5(
            cash=state.cash,
            assets=state.assets,
            work_intensity=state.work_intensity,
            energy=state.energy,
            strength=state.strength,
            work_cap=state.work_cap,
            endurance=state.endurance,
            load=state.load,
            injury=state.injury,
            time=t,
        )

        # --- Choose action ---
        if ppo_model is not None:
            obs = ppo_obs_fn(state)
            raw_action, _ = ppo_model.predict(obs, deterministic=True)
            action = ActionV5(
                invest=int(raw_action[0]),
                mode=int(raw_action[1]),
                volume=int(raw_action[2]),
                consumption=int(raw_action[3]),
            )
        else:
            action = policy_at_v5(PI, state)

        next_cash, next_assets, liq = mdp._next_financial_state(
            state.cash, state.assets, action.invest, action.consumption, t
        )
        reward = mdp._compute_reward(state, action, liq)
        val    = value_at_v5(V, state) if V is not None else float("nan")

        # Transition probability vectors
        wi_probs  = mdp.WORK_INTENSITY_MATRIX[state.work_intensity]
        en_probs  = mdp._energy_probs_formula(state.energy, state.load, state.work_intensity, action.volume, state.injury)
        ld_probs  = mdp._load_probs_formula(state.load, state.energy, state.work_intensity, action.mode, action.volume, state.injury)
        str_probs = mdp._fitness_probs_formula(state.strength, "str", state.energy, state.load, action.mode, action.volume, state.injury)
        wc_probs  = mdp._fitness_probs_formula(state.work_cap,  "wc",  state.energy, state.load, action.mode, action.volume, state.injury)
        end_probs = mdp._fitness_probs_formula(state.endurance, "end", state.energy, state.load, action.mode, action.volume, state.injury)
        inj_probs = mdp._injury_probs_formula(state.injury, state.energy, state.load, state.work_intensity, action.volume)

        rows.append({
            "month":  t + 1,
            "state":  state,
            "action": action,
            "reward": reward,
            "value":  val,
        })

        if t < mdp.time_horizon - 1:
            if rng is None:
                # Expected (argmax) path
                state = StateV5(
                    cash=next_cash,
                    assets=next_assets,
                    work_intensity=int(np.argmax(wi_probs)),
                    energy=int(np.argmax(en_probs)),
                    strength=int(np.argmax(str_probs)),
                    work_cap=int(np.argmax(wc_probs)),
                    endurance=int(np.argmax(end_probs)),
                    load=int(np.argmax(ld_probs)),
                    injury=int(np.argmax(inj_probs)),
                    time=t + 1,
                )
            else:
                # Sampled path
                state = StateV5(
                    cash=next_cash,
                    assets=next_assets,
                    work_intensity=int(rng.choice(3, p=wi_probs)),
                    energy=int(rng.choice(3, p=en_probs)),
                    strength=int(rng.choice(4, p=str_probs)),
                    work_cap=int(rng.choice(4, p=wc_probs)),
                    endurance=int(rng.choice(4, p=end_probs)),
                    load=int(rng.choice(5, p=ld_probs)),
                    injury=int(rng.choice(3, p=inj_probs)),
                    time=t + 1,
                )

    return rows


def expected_rollout(mdp, V, PI, start):
    return _rollout_core(mdp, V, PI, start, rng=None)


def sampled_rollout(mdp, V, PI, start, seed=None):
    return _rollout_core(mdp, V, PI, start, rng=np.random.default_rng(seed))


def ppo_rollout(mdp, V, PI, start, ppo_model, ppo_obs_fn, sample=False, seed=None):
    rng = np.random.default_rng(seed) if sample else None
    return _rollout_core(mdp, V, PI, start, rng=rng, ppo_model=ppo_model, ppo_obs_fn=ppo_obs_fn)


# ============================================================================
# Rich display
# ============================================================================

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel

console = Console(width=130)

INJ_STYLES = {0: "green", 1: "yellow", 2: "red"}


def build_table(rows: list[dict], mdp: MultiCapitalMDPv5, policy_label: str, mode: str) -> Table:
    nbhd = NBHD_LABELS[mdp.neighborhood]
    title = f"Policy Rollout  ·  {policy_label}  ·  Neighborhood: {nbhd}  ·  Mode: {mode}"
    tbl = Table(
        title=title,
        box=box.SIMPLE_HEAD,
        show_lines=False,
        header_style="bold white",
        title_style="bold",
    )

    # State columns
    tbl.add_column("Mo",   justify="right",  min_width=3,  no_wrap=True)
    tbl.add_column("Cash", justify="right",  min_width=7,  no_wrap=True)
    tbl.add_column("Ast",  justify="right",  min_width=6,  no_wrap=True)
    tbl.add_column("WI",   justify="center", min_width=4,  no_wrap=True)
    tbl.add_column("Eng",  justify="center", min_width=5,  no_wrap=True)
    tbl.add_column("Str",  justify="center", min_width=5,  no_wrap=True)
    tbl.add_column("WC",   justify="center", min_width=5,  no_wrap=True)
    tbl.add_column("End",  justify="center", min_width=5,  no_wrap=True)
    tbl.add_column("Load", justify="center", min_width=6,  no_wrap=True)
    tbl.add_column("Inj",  justify="center", min_width=3,  no_wrap=True)
    tbl.add_column("",     width=1,          no_wrap=True)
    # Action columns
    tbl.add_column("Mode", justify="center", min_width=4,  no_wrap=True)
    tbl.add_column("Inv",  justify="center", min_width=5,  no_wrap=True)
    tbl.add_column("Vol",  justify="center", min_width=4,  no_wrap=True)
    tbl.add_column("Spd",  justify="center", min_width=4,  no_wrap=True)
    tbl.add_column("",     width=1,          no_wrap=True)
    # Metrics
    tbl.add_column("Rew",  justify="right",  min_width=7,  no_wrap=True)
    tbl.add_column("Value",justify="right",  min_width=8,  no_wrap=True)

    for row in rows:
        s = row["state"]
        a = row["action"]
        inj_style = INJ_STYLES[s.injury]

        tbl.add_row(
            str(row["month"]),
            cash_dollars(s.cash),
            asset_dollars(s.assets),
            IW_LABELS[s.work_intensity],
            level_bar4(s.energy)[:3],        # energy is 0-2 → 3-cell bar
            level_bar4(s.strength),
            level_bar4(s.work_cap),
            level_bar4(s.endurance),
            level_bar5(s.load),
            f"[{inj_style}]{INJ_LABELS[s.injury]}[/{inj_style}]",
            "",
            f"[bold]{MODE_LABELS[a.mode]}[/bold]",
            INVEST_LABELS[a.invest],
            LEVEL_LABELS[a.volume],
            LEVEL_LABELS[a.consumption],
            "",
            f"{row['reward']:6.2f}",
            "—" if np.isnan(row["value"]) else f"{row['value']:7.1f}",
        )

    return tbl


# ── sparklines ────────────────────────────────────────────────────────────────

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def sparkline(values: list[float], lo: float = None, hi: float = None) -> str:
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    span = hi - lo if hi > lo else 1.0
    return "".join(
        _SPARK_CHARS[int((v - lo) / span * (len(_SPARK_CHARS) - 1))]
        for v in values
    )


def print_summary(rows: list[dict], mdp: MultiCapitalMDPv5, policy_label: str) -> None:
    last = rows[-1]
    s = last["state"]
    terminal_rew = mdp._terminal_reward(s)
    total_reward = sum(r["reward"] for r in rows) + terminal_rew

    inj_style = INJ_STYLES[s.injury]
    console.print(
        Panel(
            f"  Final cash: [bold]{cash_dollars(s.cash)}[/bold]"
            f"   Final assets: [bold]{asset_dollars(s.assets)}[/bold]"
            f"   Injury: [{inj_style}][bold]{INJ_LABELS[s.injury]}[/bold][/{inj_style}]\n"
            f"  Final fitness  —  Str: [bold]{s.strength}[/bold]"
            f"  WC: [bold]{s.work_cap}[/bold]"
            f"  End: [bold]{s.endurance}[/bold]"
            f"  Load: [bold]{s.load}[/bold]\n"
            f"  Terminal reward: [bold green]{terminal_rew:.1f}[/bold green]"
            f"   Cumulative step reward: {total_reward - terminal_rew:.1f}"
            f"   Total: [bold]{total_reward:.1f}[/bold]"
            + (
                f"\n  Value from t=0 (exact): [bold]{rows[0]['value']:.1f}[/bold]"
                if not np.isnan(rows[0]["value"]) else ""
            ),
            title=f"[bold]Summary — {policy_label}[/bold]",
            expand=False,
        )
    )

    console.print()
    console.print(f"[bold]Sparklines — {policy_label} (month 1 → {mdp.time_horizon})[/bold]")
    lines = [
        ("Cash    ", [r["state"].cash      for r in rows], 0, 20,  "cyan"),
        ("Assets  ", [r["state"].assets    for r in rows], 0, 30,  "blue"),
        ("Energy  ", [r["state"].energy    for r in rows], 0, 2,   "green"),
        ("Strength", [r["state"].strength  for r in rows], 0, 3,   "magenta"),
        ("WorkCap ", [r["state"].work_cap  for r in rows], 0, 3,   "bright_magenta"),
        ("Endur   ", [r["state"].endurance for r in rows], 0, 3,   "purple"),
        ("Load    ", [r["state"].load      for r in rows], 0, 4,   "red"),
        ("Injury  ", [r["state"].injury    for r in rows], 0, 2,   "bright_red"),
        ("Mode    ", [r["action"].mode     for r in rows], 0, 3,   "yellow"),
        ("Invest  ", [r["action"].invest   for r in rows], 0, 2,   "bright_yellow"),
        ("Volume  ", [r["action"].volume   for r in rows], 0, 2,   "orange3"),
        ("Spend   ", [r["action"].consumption for r in rows], 0, 2, "bright_blue"),
    ]
    for label, vals, lo, hi, color in lines:
        console.print(f"  [dim]{label}[/dim] [{color}]{sparkline(vals, lo=lo, hi=hi)}[/{color}]")
    console.print()


# ============================================================================
# Entry point
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize policy rollouts for the v5 Multi-Capital Life MDP."
    )
    # Initial state
    p.add_argument("--cash",           type=int, default=5,  help="Initial cash bin (0–20, default 5)")
    p.add_argument("--assets",         type=int, default=3,  help="Initial asset bin (0–30, default 3)")
    p.add_argument("--work-intensity", type=int, default=1,  help="Initial work intensity (0–2, default 1)")
    p.add_argument("--energy",         type=int, default=2,  help="Initial energy (0–2, default 2)")
    p.add_argument("--strength",       type=int, default=2,  help="Initial strength (0–3, default 2)")
    p.add_argument("--work-cap",       type=int, default=2,  help="Initial work capacity (0–3, default 2)")
    p.add_argument("--endurance",      type=int, default=2,  help="Initial endurance (0–3, default 2)")
    p.add_argument("--load",           type=int, default=1,  help="Initial training load (0–4, default 1)")
    p.add_argument("--injury",         type=int, default=0,  help="Initial injury (0=healthy,1=minor,2=major, default 0)")
    p.add_argument("--neighborhood",   type=int, default=1,  help="Neighborhood tier (0–2, default 1)")
    # Policy selection
    p.add_argument(
        "--policy",
        choices=["optimal", "ppo", "both"],
        default="optimal",
        help="Which policy to visualize (default: optimal)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="ppo_life_mdp_v5",
        help="Path to PPO model zip (without .zip; default: ppo_life_mdp_v5)",
    )
    # Rollout mode
    p.add_argument("--sample", action="store_true", help="Stochastic rollout instead of expected (argmax) path")
    p.add_argument("--seed",   type=int, default=None, help="RNG seed for sampled rollout")
    return p.parse_args()


def _load_ppo(model_path: str):
    from stable_baselines3 import PPO
    path = model_path if model_path.endswith(".zip") else model_path + ".zip"
    if not os.path.exists(path):
        print(f"Error: PPO model '{path}' not found.")
        sys.exit(1)
    return PPO.load(model_path)


def _make_obs_fn():
    """Return a function that converts StateV5 → obs array (no env instantiation needed)."""
    from final.gym_env_v5 import LifeGymEnv5
    _env = LifeGymEnv5.__new__(LifeGymEnv5)  # skip __init__
    return _env._state_to_obs


if __name__ == "__main__":
    args = parse_args()

    # Validate ranges
    checks = [
        ("cash",           args.cash,           0, 20),
        ("assets",         args.assets,         0, 30),
        ("work-intensity", args.work_intensity, 0, 2),
        ("energy",         args.energy,         0, 2),
        ("strength",       args.strength,       0, 3),
        ("work-cap",       args.work_cap,       0, 3),
        ("endurance",      args.endurance,      0, 3),
        ("load",           args.load,           0, 4),
        ("injury",         args.injury,         0, 2),
        ("neighborhood",   args.neighborhood,   0, 2),
    ]
    for name, val, lo, hi in checks:
        if not (lo <= val <= hi):
            print(f"Error: --{name} must be {lo}–{hi}, got {val}")
            sys.exit(1)

    mdp = MultiCapitalMDPv5(neighborhood=args.neighborhood)

    start = StateV5(
        cash=args.cash,
        assets=args.assets,
        work_intensity=args.work_intensity,
        energy=args.energy,
        strength=args.strength,
        work_cap=args.work_cap,
        endurance=args.endurance,
        load=args.load,
        injury=args.injury,
        time=0,
    )
    rollout_mode = "sampled" if args.sample else "expected"

    # Solve exactly when needed for optimal policy or value display
    V, PI = None, None
    if args.policy in ("optimal", "both"):
        print("Solving MDP (backward induction)...", end=" ", flush=True)
        V, PI = solve_mdp_fast_v5(mdp)
        save_solver_outputs_v5("exact_soln.npz", V, PI, mdp)
        print("done.")

    # ── Optimal rollout ────────────────────────────────────────────────────
    if args.policy in ("optimal", "both"):

        if args.sample:
            rows_opt = sampled_rollout(mdp, V, PI, start, seed=args.seed)
        else:
            rows_opt = expected_rollout(mdp, V, PI, start)

        console.print()
        console.print(build_table(rows_opt, mdp, "Exact Optimal", rollout_mode))
        print_summary(rows_opt, mdp, "Exact Optimal")

    # ── PPO rollout ────────────────────────────────────────────────────────
    if args.policy in ("ppo", "both"):
        ppo_model = _load_ppo(args.model)
        obs_fn    = _make_obs_fn()

        rows_ppo = ppo_rollout(
            mdp, V, PI, start, ppo_model, obs_fn,
            sample=args.sample, seed=args.seed,
        )

        console.print()
        console.print(build_table(rows_ppo, mdp, "PPO (greedy)", rollout_mode))
        print_summary(rows_ppo, mdp, "PPO (greedy)")
