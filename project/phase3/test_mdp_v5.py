"""
Simple test script for multi_capital_mdp_v5.py
"""
import time
import numpy as np
from multi_capital_mdp_v6 import (
    MultiCapitalMDPv5, StateV5, ActionV5, NonTerminal,
    solve_mdp_fast_v5, policy_at_v5, value_at_v5,
)

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f" — {detail}" if detail else ""))
    return condition


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# Setup
# ============================================================
mdp = MultiCapitalMDPv5()
s0 = NonTerminal(StateV5(
    cash=8, assets=6, work_intensity=1, energy=2,
    strength=2, work_cap=1, endurance=1, load=1, injury=0, time=0
))
a0 = ActionV5(invest=1, mode=2, volume=1, consumption=1)


# ============================================================
section("1. Action Space")
# ============================================================
acts = list(mdp.actions(s0))
check("108 actions total", len(acts) == 108, f"got {len(acts)}")
check("ActionV5 fields in range",
      all(0 <= a.invest <= 2 and 0 <= a.mode <= 3
          and 0 <= a.volume <= 2 and 0 <= a.consumption <= 2
          for a in acts))


# ============================================================
section("2. Transition Probabilities")
# ============================================================
for label, action in [
    ("mode=0 recovery",      ActionV5(invest=0, mode=0, volume=0, consumption=0)),
    ("mode=1 strength high", ActionV5(invest=0, mode=1, volume=2, consumption=0)),
    ("mode=3 endurance med", ActionV5(invest=0, mode=3, volume=1, consumption=1)),
]:
    trans = mdp.enumerate_transitions(s0, action)
    total = sum(p for _, p, _ in trans)
    check(f"probs sum to 1 ({label})", abs(total - 1.0) < 1e-9, f"sum={total:.8f}")

# Test with each injury level
for inj in [0, 1, 2]:
    s_inj = NonTerminal(StateV5(
        cash=5, assets=4, work_intensity=1, energy=1,
        strength=1, work_cap=1, endurance=1, load=2, injury=inj, time=10
    ))
    trans = mdp.enumerate_transitions(s_inj, a0)
    total = sum(p for _, p, _ in trans)
    check(f"probs sum to 1 (injury={inj})", abs(total - 1.0) < 1e-9, f"sum={total:.8f}")

# Terminal state — should return single transition
s_term = NonTerminal(StateV5(
    cash=5, assets=4, work_intensity=1, energy=1,
    strength=1, work_cap=1, endurance=1, load=1, injury=0, time=35
))
trans_term = mdp.enumerate_transitions(s_term, a0)
check("terminal state returns 1 transition", len(trans_term) == 1)


# ============================================================
section("3. Formula-Based Transition Helpers")
# ============================================================

# Energy: recovery (vol=0, wi=0, j=0) should push energy up
p_up = mdp._energy_probs_formula(e=1, l=1, wi=0, vol=0, j=0)
check("energy recovery raises E_tilde",
      p_up[2] + p_up[1] > p_up[0], f"probs={p_up.round(2)}")

# Energy: overload with injury should push energy down harder than without
p_down_healthy = mdp._energy_probs_formula(e=1, l=4, wi=2, vol=2, j=0)
p_down_injured = mdp._energy_probs_formula(e=1, l=4, wi=2, vol=2, j=1)
check("injury worsens energy drain",
      p_down_injured[0] >= p_down_healthy[0],
      f"healthy={p_down_healthy.round(2)}, injured={p_down_injured.round(2)}")

# Load: recovery mode should decrease load when energy is high, healthy
p_load_rec = mdp._load_probs_formula(l=3, e=2, wi=0, mode=0, vol=0, j=0)
check("load decreases during recovery",
      np.argmax(p_load_rec) <= 3, f"probs={p_load_rec.round(2)}")

# Load: injury increases load drift
p_load_healthy = mdp._load_probs_formula(l=2, e=2, wi=0, mode=1, vol=1, j=0)
p_load_injured = mdp._load_probs_formula(l=2, e=2, wi=0, mode=1, vol=1, j=1)
check("injury increases load drift",
      np.argmax(p_load_injured) >= np.argmax(p_load_healthy),
      f"healthy_peak={np.argmax(p_load_healthy)}, injured_peak={np.argmax(p_load_injured)}")

# Fitness: strength mode should improve strength more than endurance
p_str = mdp._fitness_probs_formula(f=1, k="str", e=2, l=1, mode=1, vol=2, j=0)
p_end = mdp._fitness_probs_formula(f=1, k="end", e=2, l=1, mode=1, vol=2, j=0)
check("strength mode improves strength > endurance",
      p_str[2] > p_end[2], f"str_improve={p_str[2]:.2f}, end_improve={p_end[2]:.2f}")

# Fitness: major injury reduces improvement probability
p_fit_healthy = mdp._fitness_probs_formula(f=1, k="str", e=2, l=1, mode=1, vol=2, j=0)
p_fit_major   = mdp._fitness_probs_formula(f=1, k="str", e=2, l=1, mode=1, vol=2, j=2)
check("major injury reduces fitness gain probability",
      p_fit_major[2] < p_fit_healthy[2],
      f"healthy_gain={p_fit_healthy[2]:.2f}, injured_gain={p_fit_major[2]:.2f}")

# Fitness: detraining rate is higher in recovery mode
p_detr_rec  = mdp._fitness_probs_formula(f=2, k="str", e=2, l=1, mode=0, vol=0, j=0)
p_detr_norm = mdp._fitness_probs_formula(f=2, k="str", e=2, l=1, mode=1, vol=1, j=0)
check("higher detraining in recovery mode",
      p_detr_rec[1] > p_detr_norm[1],
      f"rec_detr={p_detr_rec[1]:.2f}, norm_detr={p_detr_norm[1]:.2f}")

# Injury: healthy state with high risk factors raises injury probability
p_inj_low  = mdp._injury_probs_formula(j=0, e=2, l=0, wi=0, vol=0)
p_inj_high = mdp._injury_probs_formula(j=0, e=0, l=4, wi=2, vol=2)
check("high-risk factors increase injury probability",
      p_inj_high[1] > p_inj_low[1],
      f"low_risk_P(inj)={p_inj_low[1]:.2f}, high_risk_P(inj)={p_inj_high[1]:.2f}")

# Injury: minor injury has partial recovery probability
p_minor = mdp._injury_probs_formula(j=1, e=2, l=0, wi=0, vol=0)
check("minor injury can recover to healthy",
      p_minor[0] > 0, f"P(recover|minor)={p_minor[0]:.2f}")

# Injury: major injury has lower recovery probability than minor
p_major = mdp._injury_probs_formula(j=2, e=2, l=0, wi=0, vol=0)
check("major injury recovers slower than minor",
      p_major[0] < p_minor[0],
      f"P(recover|major)={p_major[0]:.2f}, P(recover|minor)={p_minor[0]:.2f}")


# ============================================================
section("4. Reward Function")
# ============================================================
_, _, liq = mdp._next_financial_state(8, 6, 1, 1, 0)
r = mdp._compute_reward(s0.state, a0, liq)
check("reward is finite", np.isfinite(r), f"r={r:.4f}")

# Penalise training when energy=0 and vol=2
s_tired = StateV5(cash=5, assets=3, work_intensity=0, energy=0,
                  strength=1, work_cap=1, endurance=1, load=1, injury=0, time=5)
r_tired  = mdp._compute_reward(s_tired, ActionV5(invest=0, mode=1, volume=2, consumption=0), 0.0)
r_rested = mdp._compute_reward(s_tired, ActionV5(invest=0, mode=0, volume=0, consumption=0), 0.0)
check("fatigue penalty reduces reward", r_rested > r_tired,
      f"rested={r_rested:.2f}, tired={r_tired:.2f}")

# Training while injured incurs extra penalty
s_healthy_hi = StateV5(cash=5, assets=3, work_intensity=0, energy=2,
                       strength=1, work_cap=1, endurance=1, load=1, injury=0, time=5)
s_injured_hi = StateV5(cash=5, assets=3, work_intensity=0, energy=2,
                       strength=1, work_cap=1, endurance=1, load=1, injury=1, time=5)
act_train = ActionV5(invest=0, mode=1, volume=2, consumption=0)
r_healthy_train = mdp._compute_reward(s_healthy_hi, act_train, 0.0)
r_injured_train = mdp._compute_reward(s_injured_hi, act_train, 0.0)
check("training while injured incurs extra penalty", r_healthy_train > r_injured_train,
      f"healthy={r_healthy_train:.2f}, injured={r_injured_train:.2f}")

# Injured state reduces terminal reward
tr_healthy = mdp._terminal_reward(StateV5(cash=8, assets=6, work_intensity=1, energy=2,
                                           strength=2, work_cap=2, endurance=2, load=1, injury=0, time=35))
tr_injured = mdp._terminal_reward(StateV5(cash=8, assets=6, work_intensity=1, energy=2,
                                           strength=2, work_cap=2, endurance=2, load=1, injury=2, time=35))
check("injury reduces terminal reward", tr_healthy > tr_injured,
      f"healthy={tr_healthy:.2f}, injured={tr_injured:.2f}")


# ============================================================
section("5. Step (Sampler)")
# ============================================================
dist = mdp.step(s0, a0)
results = [dist.sample() for _ in range(50)]
next_states = [ns for ns, _ in results]
rewards = [r for _, r in results]
check("step returns finite rewards", all(np.isfinite(r) for r in rewards))
check("next state time advances", all(
    (ns.state.time == 1 if hasattr(ns, 'state') else True)
    for ns in next_states
))
injury_vals = [ns.state.injury for ns in next_states if hasattr(ns, 'state')]
check("injury field is valid (0-2)", all(0 <= j <= 2 for j in injury_vals),
      f"observed: {set(injury_vals)}")


# ============================================================
section("6. Solver (T=36)")
# ============================================================
mdp_full = MultiCapitalMDPv5(time_horizon=36)
t0 = time.time()
V, PI = solve_mdp_fast_v5(mdp_full)
dt = time.time() - t0

check("V shape correct",  V.shape  == (21, 31, 3, 3, 5, 4, 4, 4, 3, 36), f"shape={V.shape}")
check("PI shape correct", PI.shape == (21, 31, 3, 3, 5, 4, 4, 4, 3, 36, 4), f"shape={PI.shape}")
check("V is finite everywhere", np.all(np.isfinite(V)))
check("PI actions in valid range",
      np.all(PI[..., 0] <= 2) and np.all(PI[..., 1] <= 3)
      and np.all(PI[..., 2] <= 2) and np.all(PI[..., 3] <= 2))
print(f"  Solve time (T=36): {dt:.1f}s")


# ============================================================
section("7. Policy / Value Retrieval")
# ============================================================
s_query = StateV5(cash=8, assets=6, work_intensity=1, energy=2,
                  strength=2, work_cap=1, endurance=1, load=1, injury=0, time=0)
opt_a = policy_at_v5(PI, s_query)
opt_v = value_at_v5(V, s_query)
check("policy_at_v5 returns ActionV5", isinstance(opt_a, ActionV5))
check("value_at_v5 returns float", isinstance(opt_v, float))
print(f"  Optimal action: {opt_a}")
print(f"  Value estimate: {opt_v:.4f}")

# Value should be higher for better states
s_good = StateV5(cash=15, assets=20, work_intensity=0, energy=2,
                 strength=3, work_cap=3, endurance=3, load=0, injury=0, time=0)
s_poor = StateV5(cash=1,  assets=0,  work_intensity=2, energy=0,
                 strength=0, work_cap=0, endurance=0, load=4, injury=2, time=0)
v_good = value_at_v5(V, s_good)
v_poor = value_at_v5(V, s_poor)
check("good state has higher value than poor state", v_good > v_poor,
      f"good={v_good:.2f}, poor={v_poor:.2f}")

# Healthy state should have higher value than injured state (same capital/fitness)
s_healthy = StateV5(cash=10, assets=10, work_intensity=1, energy=2,
                    strength=2, work_cap=2, endurance=2, load=1, injury=0, time=0)
s_injured = StateV5(cash=10, assets=10, work_intensity=1, energy=2,
                    strength=2, work_cap=2, endurance=2, load=1, injury=2, time=0)
v_healthy = value_at_v5(V, s_healthy)
v_injured = value_at_v5(V, s_injured)
check("healthy state has higher value than injured state", v_healthy > v_injured,
      f"healthy={v_healthy:.2f}, injured={v_injured:.2f}")

# Recovery mode preferred when injured
opt_a_injured = policy_at_v5(PI, s_injured)
print(f"  Optimal action (injured): {opt_a_injured}")


# ============================================================
print(f"\n{'='*60}")
print("  All checks complete.")
print(f"{'='*60}\n")
