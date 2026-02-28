"""
Comprehensive Sensitivity Analysis — Multi-Capital Energy MDP

Three-tier sweep design:
  Tier 1: Reward weights  (8 single-parameter + 4 two-parameter interaction grids)
  Tier 2: Transition dynamics  (work persistence, pressure knobs, perf degradation)
  Tier 3: Economics  (rentxsalary grid, invest returnxcost grid)
  Global: Random sampling over 10 parameters (N=150), Pearson importance ranking
  Disc:   Discretization robustness (11 vs 21 wealth bins, 5 key sweeps)

Saved figures:
  sensitivity_1d.png        -- Tier 1 single-parameter sweeps
  sensitivity_2d.png        -- Tier 1 two-parameter interaction grids
  sensitivity_dynamics.png  -- Tier 2 dynamics sweeps
  sensitivity_economics.png -- Tier 3 economics grids
  sensitivity_global.png    -- Global sensitivity (correlation heatmap + importance)
  sensitivity_disc.png      -- Discretization robustness check

Run:  python sensitivity_analysis.py
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from typing import List, Dict, Tuple, Optional

from multi_capital_mdp import (
    MultiCapitalEnergyMDP, StateV3,
    solve_mdp_fast, value_at, policy_at,
)


# ========================================= ===================================
# Constants
# ============================================================================

S0 = StateV3(wealth=2, location=1, work_intensity=1, energy=2, performance=1, time=0)
DEFAULT_KWARGS = dict(gamma=0.99, time_horizon=36)

# 24-state distribution for mean/worst-case V (realistic starting conditions)
INIT_STATES = [
    StateV3(wealth=w, location=1, work_intensity=wi, energy=en, performance=1, time=0)
    for w  in range(1, 5)    # bins 1-4  ($20k – $80k)
    for wi in range(3)        # all work-intensity levels
    for en in [1, 2]          # medium and high energy
]


# ============================================================================
# Section 1 — Metrics
# ============================================================================

def action_fractions(PI: np.ndarray, t: int = 0) -> dict:
    """
    Over all states at time t, compute the fraction choosing each action level.
    Returns dict with keys 'invest', 'consume', 'train', 'housing',
    each a list of 3 fractions summing to 1.
    """
    PI_t = PI[:, :, :, :, :, t, :]   # (NW, 3, 3, 3, 3, 4)
    h    = PI_t[..., 0] - 1           # back to {-1,0,1}
    inv  = PI_t[..., 1]
    tr   = PI_t[..., 2]
    c    = PI_t[..., 3]
    return {
        'invest':  [float(np.mean(inv == i)) for i in range(3)],
        'consume': [float(np.mean(c   == i)) for i in range(3)],
        'train':   [float(np.mean(tr  == i)) for i in range(3)],
        'housing': [float(np.mean(h   == i)) for i in [-1, 0, 1]],
    }


def value_distribution(V: np.ndarray, states: list = INIT_STATES) -> Tuple[float, float]:
    """Mean and worst-case V over the given distribution of initial states."""
    vals = [value_at(V, s) for s in states]
    return float(np.mean(vals)), float(np.min(vals))


def simulate_metrics(
    mdp: MultiCapitalEnergyMDP,
    PI:  np.ndarray,
    s0:  StateV3 = S0,
    n:   int     = 300,
    seed: int    = 0,
) -> dict:
    """
    Vectorized batch Monte Carlo: n parallel episodes from s0.

    Returns:
        mean_low_energy:    fraction of time steps with energy == 0
        peak_success_rate:  fraction of peak months (t%3==2) with performance == 2
    """
    rng = np.random.RandomState(seed)
    T   = mdp.time_horizon
    NW  = mdp.n_wealth_bins
    WU  = mdp.wealth_unit

    # Lookup arrays for deterministic transitions
    rents_arr = np.array(mdp._rents)                         # (3,)
    spend_arr = np.array(mdp._spend)                         # (3,)
    icost_arr = np.array(mdp.invest_costs) * mdp._salary_base # (3,) — invest cost in $/month
    iret_arr  = np.array(mdp.invest_returns) * WU            # (3,) — return per wealth-unit

    # Stochastic transition tables
    wi_mat    = mdp.work_intensity_matrix                     # (3, 3)
    en_arr    = np.array([mdp.ENERGY_TABLE[e] for e in range(3)])  # (3, 5, 3)
    perf_arr  = np.array([
        [[mdp.PERFORMANCE_TABLE[ph][p][q] for q in range(3)] for p in range(3)]
        for ph in range(3)
    ])  # (3, 3, 3, 3) — [phase, perf, quality, perf']

    def _sample(probs: np.ndarray) -> np.ndarray:
        """Vectorized inverse-CDF sampling.  probs: (n,3) → (n,) in {0,1,2}."""
        cdf = np.cumsum(probs, axis=1)
        cdf[:, -1] = 1.0  # numerical stability
        u = rng.random(n)
        return np.clip((u[:, None] > cdf).sum(axis=1), 0, 2).astype(int)

    # Initialise episode batch
    w  = np.full(n, s0.wealth,         dtype=int)
    l  = np.full(n, s0.location,       dtype=int)
    wi = np.full(n, s0.work_intensity, dtype=int)
    en = np.full(n, s0.energy,         dtype=int)
    pf = np.full(n, s0.performance,    dtype=int)

    low_en_total = 0
    peak_success = 0
    peak_total   = 0

    for t in range(T - 1):
        # Accumulate BEFORE transition (reward at current state)
        low_en_total += int(np.sum(en == 0))
        if t % 3 == 2:
            peak_success += int(np.sum(pf == 2))
            peak_total   += n

        # Look up optimal actions (fancy indexing over batch)
        raw   = PI[w, l, wi, en, pf, t, :]    # (n, 4)
        h_raw = raw[:, 0] - 1                  # {-1,0,1}
        inv   = raw[:, 1]
        tr    = raw[:, 2]
        c     = raw[:, 3]

        # Deterministic transitions
        l_new  = np.clip(l + h_raw, 0, 2).astype(int)
        salary = mdp._salary_base * (mdp.R_RAISE ** (t // 12))
        delta  = (
            salary
            - rents_arr[l_new]
            - spend_arr[c]
            - icost_arr[inv]
            + iret_arr[inv] * w
        )
        w_new = np.clip(np.round(w + delta / WU).astype(int), 0, NW - 1)

        # wi' — vectorised
        wi_new = _sample(wi_mat[wi])

        # en' — pressure -> table row -> vectorised
        loc_bonus  = (l_new == 2).astype(int)
        pressure   = np.clip(np.round(
            mdp.pressure_work_coef  * wi
            + mdp.pressure_train_coef * tr
            - mdp.pressure_loc_coef   * loc_bonus
        ).astype(int), 0, 4)
        en_new = _sample(en_arr[en, 4 - pressure, :])

        # perf' — quality -> table -> vectorised
        quality = np.maximum(
            np.minimum(tr, en) - mdp.perf_work_penalty * (wi == 2).astype(int), 0
        )
        pf_new = _sample(perf_arr[t % 3, pf, quality, :])

        w, l, wi, en, pf = w_new, l_new, wi_new, en_new, pf_new

    # Metrics at final step
    low_en_total += int(np.sum(en == 0))
    if (T - 1) % 3 == 2:
        peak_success += int(np.sum(pf == 2))
        peak_total   += n

    return {
        'mean_low_energy':   float(low_en_total) / (T * n),
        'peak_success_rate': float(peak_success) / peak_total if peak_total > 0 else 0.0,
    }


# ============================================================================
# Section 2 — Sweep runners
# ============================================================================

def _policy_slice(PI: np.ndarray, NW: int, t: int = 0,
                  l: int = 1, wi: int = 1, en: int = 2, perf: int = 1) -> dict:
    """Policy at a fixed non-wealth slice (loc=1, wi=1, en=2, perf=1) across wealth bins."""
    raw = PI[range(NW), l, wi, en, perf, t, :]  # (NW, 4)
    return {
        'housing': raw[:, 0] - 1,
        'invest':  raw[:, 1],
        'train':   raw[:, 2],
        'consume': raw[:, 3],
    }


def solve_and_metrics(mdp_kwargs: dict, s0: StateV3 = S0, n_sim: int = 300) -> dict:
    """Solve one MDP and return a full metric dictionary."""
    mdp = MultiCapitalEnergyMDP(**{**DEFAULT_KWARGS, **mdp_kwargs})
    V, PI = solve_mdp_fast(mdp)
    mean_v, worst_v = value_distribution(V)
    fracs   = action_fractions(PI)
    sim     = simulate_metrics(mdp, PI, s0, n=n_sim)
    pslice  = _policy_slice(PI, mdp.n_wealth_bins)
    return {
        'mdp':     mdp,
        'V':       V,
        'PI':      PI,
        'V0':      value_at(V, s0),
        'mean_V':  mean_v,
        'worst_V': worst_v,
        'fracs':   fracs,
        'sim':     sim,
        'pslice':  pslice,
    }


def run_1d_sweep(sweep_def: dict, n_sim: int = 300) -> list:
    """Run a 1D parameter sweep. Returns list of metric dicts."""
    run_list = []
    for v in sweep_def['values']:
        m = solve_and_metrics(sweep_def['kwargs'](v), n_sim=n_sim)
        m['param_value'] = v
        run_list.append(m)
    return run_list


def run_2d_sweep(grid_def: dict, n_sim: int = 200) -> dict:
    """
    Run a 2D parameter interaction grid.
    Returns grid_def enriched with 'grid': [[metric_dict for v2 in p2_vals] for v1 in p1_vals].
    """
    grid = []
    for v1 in grid_def['p1_values']:
        row = []
        for v2 in grid_def['p2_values']:
            kw = {**grid_def['p1_kwargs'](v1), **grid_def['p2_kwargs'](v2)}
            row.append(solve_and_metrics(kw, n_sim=n_sim))
        grid.append(row)
    return {**grid_def, 'grid': grid}


# ============================================================================
# Section 3 — Sweep definitions
# ============================================================================

_BR = MultiCapitalEnergyMDP.RENTS  # base rent dict for Tier 3

TIER1_SWEEPS = [
    {'name': 'beta_peak',         'label': 'β (peak bonus)',
     'values': [0, 3, 7, 10, 15, 20],
     'kwargs': lambda v: {'beta_peak': v}},
    {'name': 'alpha_performance', 'label': 'alpha_perf',
     'values': [0.5, 1, 2, 3, 5, 8],
     'kwargs': lambda v: {'alpha_performance': v}},
    {'name': 'alpha_energy',      'label': 'alpha_energy',
     'values': [0.5, 1, 2, 3, 5],
     'kwargs': lambda v: {'alpha_energy': v}},
    {'name': 'alpha_wealth',      'label': 'alpha_wealth',
     'values': [0.5, 1, 2, 4, 6],
     'kwargs': lambda v: {'alpha_wealth': v}},
    {'name': 'alpha_location',    'label': 'alpha_loc',
     'values': [0.5, 1, 1.5, 3, 5],
     'kwargs': lambda v: {'alpha_location': v}},
    {'name': 'alpha_distress',    'label': 'alpha_distress',
     'values': [2, 4, 8, 12, 16],
     'kwargs': lambda v: {'alpha_distress': v}},
    {'name': 'alpha_consumption', 'label': 'alpha_cons',
     'values': [0, 0.5, 1, 2, 3],
     'kwargs': lambda v: {'alpha_consumption': v}},
    {'name': 'invest_return',     'label': 'r_invest max',
     'values': [0.004, 0.008, 0.012, 0.020, 0.030],
     'kwargs': lambda v: {'invest_returns': (round(v/6, 6), round(v/2, 6), v)}},
]

TIER1_GRIDS = [
    {'title': 'β_peak x alpha_perf',
     'story': 'Peak incentive needs readiness value',
     'p1_name': 'beta_peak',         'p1_label': 'β',
     'p1_values': [0, 5, 10, 15, 20], 'p1_kwargs': lambda v: {'beta_peak': v},
     'p2_name': 'alpha_performance', 'p2_label': 'alpha_perf',
     'p2_values': [1, 2, 3, 5, 8],   'p2_kwargs': lambda v: {'alpha_performance': v}},
    {'title': 'alpha_wealth x alpha_cons',
     'story': 'Save vs enjoy life',
     'p1_name': 'alpha_wealth',        'p1_label': 'alpha_wealth',
     'p1_values': [0.5, 1, 2, 4, 6],  'p1_kwargs': lambda v: {'alpha_wealth': v},
     'p2_name': 'alpha_consumption',   'p2_label': 'alpha_cons',
     'p2_values': [0, 0.5, 1, 2, 3],  'p2_kwargs': lambda v: {'alpha_consumption': v}},
    {'title': 'alpha_energy x alpha_perf',
     'story': 'Burn energy vs manage it',
     'p1_name': 'alpha_energy',        'p1_label': 'alpha_energy',
     'p1_values': [0.5, 1, 2, 3, 5],  'p1_kwargs': lambda v: {'alpha_energy': v},
     'p2_name': 'alpha_performance',   'p2_label': 'alpha_perf',
     'p2_values': [1, 2, 3, 5, 8],    'p2_kwargs': lambda v: {'alpha_performance': v}},
    {'title': 'alpha_loc x alpha_wealth',
     'story': 'Rent trap unless wealth is valued',
     'p1_name': 'alpha_location',      'p1_label': 'alpha_loc',
     'p1_values': [0.5, 1, 1.5, 3, 5], 'p1_kwargs': lambda v: {'alpha_location': v},
     'p2_name': 'alpha_wealth',         'p2_label': 'alpha_wealth',
     'p2_values': [0.5, 1, 2, 4, 6],   'p2_kwargs': lambda v: {'alpha_wealth': v}},
]

TIER2_DYN = [
    {'name': 'wi_persistence',      'label': 'wi_persist',
     'values': [0.4, 0.5, 0.6, 0.7, 0.8],
     'kwargs': lambda v: {'wi_persistence': v}},
    {'name': 'pressure_work_coef',  'label': 'p_work',
     'values': [0.5, 1.0, 1.5, 2.0],
     'kwargs': lambda v: {'pressure_work_coef': v}},
    {'name': 'pressure_train_coef', 'label': 'p_train',
     'values': [0.5, 1.0, 1.5, 2.0],
     'kwargs': lambda v: {'pressure_train_coef': v}},
    {'name': 'perf_work_penalty',   'label': 'perf_pen',
     'values': [0, 1, 2],
     'kwargs': lambda v: {'perf_work_penalty': v}},
]

TIER3_ECON = [
    {'title': 'rent_mult x salary_mult',
     'story': 'Can I afford a good neighborhood?',
     'p1_name': 'rent_mult',   'p1_label': 'rent mult',
     'p1_values': [0.7, 1.0, 1.3, 1.6],
     'p1_kwargs': lambda v: {'rent_levels': (_BR[0]*v, _BR[1]*v, _BR[2]*v)},
     'p2_name': 'salary_mult', 'p2_label': 'salary mult',
     'p2_values': [0.7, 1.0, 1.3, 1.6],
     'p2_kwargs': lambda v: {'salary_mult': v}},
    {'title': 'max_invest_return x max_invest_cost',
     'story': 'When does aggressive investing pay off?',
     'p1_name': 'max_invest_return', 'p1_label': 'r_invest max',
     'p1_values': [0.006, 0.012, 0.020, 0.030],
     'p1_kwargs': lambda v: {'invest_returns': (round(v/6, 6), round(v/2, 6), v)},
     'p2_name': 'max_invest_cost',   'p2_label': 'cost max',
     'p2_values': [0.04, 0.07, 0.09, 0.12],
     'p2_kwargs': lambda v: {'invest_costs': (0.0, round(v/2, 4), v)}},
]


# ============================================================================
# Section 4 — Global sensitivity
# ============================================================================

GLOBAL_PARAMS = {
    'beta_peak':          (0.0,  20.0),
    'alpha_performance':  (0.5,   8.0),
    'alpha_energy':       (0.5,   5.0),
    'alpha_wealth':       (0.2,   6.0),
    'alpha_location':     (0.3,   5.0),
    'alpha_distress':     (2.0,  16.0),
    'alpha_consumption':  (0.0,   4.0),
    'wi_persistence':     (0.4,   0.85),
    'max_invest_return':  (0.004, 0.030),
    'max_invest_cost':    (0.02,  0.15),
}


def run_global_sensitivity(N: int = 150, seed: int = 1, n_sim: int = 200) -> dict:
    """
    Sample N random parameter configurations, solve each MDP, and return
    a parameter matrix and metric matrix for correlation analysis.
    """
    rng = np.random.RandomState(seed)
    param_names = list(GLOBAL_PARAMS.keys())
    metric_names = ['V0', 'mean_V', 'peak_success', 'low_energy', 'invest2%', 'consume2%']

    param_matrix  = np.zeros((N, len(param_names)))
    metric_matrix = np.zeros((N, len(metric_names)))

    for i in range(N):
        # Sample each parameter uniformly in its range
        row = {}
        for k, (lo, hi) in GLOBAL_PARAMS.items():
            row[k] = float(rng.uniform(lo, hi))
        param_matrix[i] = [row[k] for k in param_names]

        # Build MDP kwargs
        v_r = row['max_invest_return']
        v_c = row['max_invest_cost']
        kw = {k: row[k] for k in param_names if k not in ('max_invest_return', 'max_invest_cost')}
        kw['invest_returns'] = (round(v_r/6, 6), round(v_r/2, 6), v_r)
        kw['invest_costs']   = (0.0, round(v_c/2, 4), v_c)

        m = solve_and_metrics(kw, n_sim=n_sim)
        metric_matrix[i] = [
            m['V0'],
            m['mean_V'],
            m['sim']['peak_success_rate'],
            m['sim']['mean_low_energy'],
            m['fracs']['invest'][2],
            m['fracs']['consume'][2],
        ]

        if (i + 1) % 25 == 0:
            print(f"    Global: {i+1}/{N} complete")

    # Pearson correlations: (n_params, n_metrics)
    corr = np.zeros((len(param_names), len(metric_names)))
    for j, pname in enumerate(param_names):
        for k, mname in enumerate(metric_names):
            x, y = param_matrix[:, j], metric_matrix[:, k]
            if x.std() > 0 and y.std() > 0:
                corr[j, k] = float(np.corrcoef(x, y)[0, 1])

    return {
        'param_names':   param_names,
        'metric_names':  metric_names,
        'param_matrix':  param_matrix,
        'metric_matrix': metric_matrix,
        'corr':          corr,
    }


# ============================================================================
# Section 5 — Discretization robustness check
# ============================================================================

DISC_SWEEPS = ['beta_peak', 'alpha_performance', 'alpha_wealth',
               'alpha_consumption', 'invest_return']   # 5 key sweeps


def run_disc_check(n_sim: int = 150) -> dict:
    """
    Re-run 5 Tier 1 sweeps with n_wealth_bins=21 (wealth_unit=$10k).
    Returns parallel results alongside the standard 11-bin results.
    """
    results = {}
    tier1_by_name = {s['name']: s for s in TIER1_SWEEPS}
    for name in DISC_SWEEPS:
        sweep_def = tier1_by_name[name]
        run_11 = run_1d_sweep(sweep_def, n_sim=n_sim)
        # 21-bin version: add n_wealth_bins=21 to every mdp_kwargs call
        def _wrap(v, sd=sweep_def):
            kw = sd['kwargs'](v)
            kw['n_wealth_bins'] = 21
            return kw
        run_21 = []
        for v in sweep_def['values']:
            m = solve_and_metrics(_wrap(v), n_sim=n_sim)
            m['param_value'] = v
            run_21.append(m)
        results[name] = {'sweep_def': sweep_def, 'run_11': run_11, 'run_21': run_21}
    return results


# ============================================================================
# Section 6 — Terminal display
# ============================================================================

def _metrics_row(run_list: list) -> None:
    """Print a compact metrics block for a 1D sweep."""
    vals = [r['param_value'] for r in run_list]
    hdr  = '  '.join(f"{v:>8.3g}" for v in vals)
    print(f"\n  {'Metric':>22}  {hdr}")
    print(f"  {'-'*22}  " + '  '.join(['-'*8]*len(vals)))

    rows = [
        ('V(s0)',            [f"{r['V0']:>8.1f}"                         for r in run_list]),
        ('mean V(init)',     [f"{r['mean_V']:>8.1f}"                     for r in run_list]),
        ('worst V(init)',    [f"{r['worst_V']:>8.1f}"                    for r in run_list]),
        ('peak success',     [f"{r['sim']['peak_success_rate']:>8.2f}"   for r in run_list]),
        ('low energy frac',  [f"{r['sim']['mean_low_energy']:>8.2f}"     for r in run_list]),
        ('invest=2 frac',    [f"{r['fracs']['invest'][2]:>8.2f}"         for r in run_list]),
        ('consume=2 frac',   [f"{r['fracs']['consume'][2]:>8.2f}"        for r in run_list]),
        ('train=2 frac',     [f"{r['fracs']['train'][2]:>8.2f}"          for r in run_list]),
    ]
    for label, cells in rows:
        print(f"  {label:>22}  " + '  '.join(cells))


def print_1d_table(sweep_def: dict, run_list: list) -> None:
    """Print the policy table and metrics block for one 1D sweep."""
    values = [r['param_value'] for r in run_list]
    label  = sweep_def['label']

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")

    val_strs = [f"{v:>8.3g}" for v in values]
    print(f"  {'Wealth bin':>16}  " + '  '.join(val_strs))
    print(f"  {'':>16}  " + '  '.join([' h/i/tr/c' for _ in val_strs]))
    print(f"  {'-'*16}  " + '  '.join(['-'*8]*len(val_strs)))

    NW = run_list[0]['mdp'].n_wealth_bins
    WU = run_list[0]['mdp'].wealth_unit
    wu_k = int(WU / 1000)

    for w in range(NW):
        cells = []
        for r in run_list:
            ps = r['pslice']
            h  = {-1:'dn', 0:'--', 1:'up'}[int(ps['housing'][w])]
            iv = str(int(ps['invest'][w]))
            tr = str(int(ps['train'][w]))
            c  = str(int(ps['consume'][w]))
            cells.append(f"{h}/{iv}/{tr}/{c}")
        wlabel = f"bin {w:>2} (${w*wu_k:>4}k)"
        print(f"  {wlabel:>16}  " + '  '.join(f"{cell:>8}" for cell in cells))

    _metrics_row(run_list)


# ============================================================================
# Section 7 — Figures
# ============================================================================

METRIC_KEYS = ['V0', 'mean_V', 'peak_success_rate', 'low_energy', 'invest2%', 'consume2%']
METRIC_LABELS = ['V(s₀)', 'mean V', 'peak success', 'low energy', 'invest=2%', 'consume=2%']
METRIC_COLORS = ['steelblue', 'slateblue', 'forestgreen', 'tomato', 'darkorange', 'mediumpurple']


def _extract(run_list: list, metric: str) -> np.ndarray:
    out = []
    for r in run_list:
        if   metric == 'V0':             out.append(r['V0'])
        elif metric == 'mean_V':         out.append(r['mean_V'])
        elif metric == 'worst_V':        out.append(r['worst_V'])
        elif metric == 'peak_success_rate': out.append(r['sim']['peak_success_rate'])
        elif metric == 'low_energy':     out.append(r['sim']['mean_low_energy'])
        elif metric == 'invest2%':       out.append(r['fracs']['invest'][2])
        elif metric == 'consume2%':      out.append(r['fracs']['consume'][2])
        elif metric == 'invest_heatmap': out.append(r['pslice']['invest'])
        elif metric == 'consume_heatmap':out.append(r['pslice']['consume'])
    return np.array(out)


def plot_1d_figure(sweeps_defs: list, run_lists: list, filename: str) -> None:
    """
    Multi-panel figure for 1D sweeps.
    Rows: V(s0), mean V, peak success, invest-level heatmap, consume-level heatmap.
    Columns: one per sweep.
    """
    n_sweeps = len(sweeps_defs)
    metrics_top = [('V(s₀)', 'V0', 'steelblue'),
                   ('mean V(init)', 'mean_V', 'slateblue'),
                   ('peak success', 'peak_success_rate', 'forestgreen')]
    n_rows = len(metrics_top) + 2  # + invest heatmap + consume heatmap

    fig, axes = plt.subplots(n_rows, n_sweeps, figsize=(3.8 * n_sweeps, 3.5 * n_rows))
    if n_sweeps == 1:
        axes = axes[:, None]
    fig.suptitle('Sensitivity Analysis — Tier 1 Single-Parameter Sweeps', fontsize=13, y=1.01)

    NW_default = 11
    WEALTH_LABELS = [f"${w*20}k" for w in range(NW_default)]

    for col, (sweep_def, run_list) in enumerate(zip(sweeps_defs, run_lists)):
        values = [r['param_value'] for r in run_list]
        label  = sweep_def['label']

        # Line plots (top 3 rows)
        for row, (mlabel, mkey, color) in enumerate(metrics_top):
            ax = axes[row, col]
            y  = _extract(run_list, mkey)
            ax.plot(values, y, 'o-', color=color, lw=2, ms=5)
            ax.set_title(f"{mlabel}\nvs {label}", fontsize=8)
            ax.set_xlabel(label, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(mlabel, fontsize=8)

        # Invest heatmap (row 3)
        ax = axes[n_rows - 2, col]
        inv_mat = _extract(run_list, 'invest_heatmap').T   # (NW, n_values)
        im = ax.imshow(inv_mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=2, origin='lower')
        ax.set_title(f"Invest level\nvs {label}", fontsize=8)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{v:.3g}" for v in values], fontsize=6, rotation=30)
        if inv_mat.shape[0] == NW_default:
            ax.set_yticks(range(NW_default))
            ax.set_yticklabels(WEALTH_LABELS, fontsize=5)
        plt.colorbar(im, ax=ax, ticks=[0, 1, 2], label='invest', fraction=0.046)

        # Consume heatmap (row 4)
        ax = axes[n_rows - 1, col]
        c_mat = _extract(run_list, 'consume_heatmap').T    # (NW, n_values)
        im2 = ax.imshow(c_mat, aspect='auto', cmap='Oranges', vmin=0, vmax=2, origin='lower')
        ax.set_title(f"Consume level\nvs {label}", fontsize=8)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{v:.3g}" for v in values], fontsize=6, rotation=30)
        if c_mat.shape[0] == NW_default:
            ax.set_yticks(range(NW_default))
            ax.set_yticklabels(WEALTH_LABELS, fontsize=5)
        plt.colorbar(im2, ax=ax, ticks=[0, 1, 2], label='consume', fraction=0.046)

    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_2d_figure(grid_results: list, filename: str) -> None:
    """
    4-panel per grid: V(s0), peak success, low energy, invest=2 fraction.
    Layout: n_grids rows x 4 cols.
    """
    n_grids  = len(grid_results)
    n_panels = 4
    fig, axes = plt.subplots(n_grids, n_panels, figsize=(4.5 * n_panels, 4 * n_grids))
    if n_grids == 1:
        axes = axes[None, :]
    fig.suptitle('Sensitivity Analysis — Two-Parameter Interaction Grids', fontsize=13, y=1.01)

    panel_defs = [
        ('V(s₀)',        lambda r: r['V0'],                             'Blues',    None, None),
        ('peak success', lambda r: r['sim']['peak_success_rate'],        'Greens',   0, 1),
        ('low energy',   lambda r: r['sim']['mean_low_energy'],          'Reds',     0, 0.4),
        ('invest=2 frac',lambda r: r['fracs']['invest'][2],              'Oranges',  0, 1),
    ]

    for gidx, gr in enumerate(grid_results):
        p1v = gr['p1_values']
        p2v = gr['p2_values']

        for pidx, (plab, extfn, cmap, vmin, vmax) in enumerate(panel_defs):
            ax = axes[gidx, pidx]
            mat = np.array([[extfn(gr['grid'][i][j])
                             for j in range(len(p2v))]
                            for i in range(len(p1v))])
            kw = {'aspect': 'auto', 'cmap': cmap, 'origin': 'lower'}
            if vmin is not None:
                kw.update(vmin=vmin, vmax=vmax)
            im = ax.imshow(mat, **kw)
            ax.set_title(f"{gr['title']}\n{plab}", fontsize=8)
            ax.set_xlabel(gr['p2_label'], fontsize=7)
            if pidx == 0:
                ax.set_ylabel(gr['p1_label'], fontsize=7)
            ax.set_xticks(range(len(p2v)))
            ax.set_xticklabels([f"{v:.3g}" for v in p2v], fontsize=6, rotation=30)
            ax.set_yticks(range(len(p1v)))
            ax.set_yticklabels([f"{v:.3g}" for v in p1v], fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Add story annotation
        axes[gidx, 0].text(-0.5, -0.18, f"Story: {gr['story']}", fontsize=7,
                            transform=axes[gidx, 0].transAxes, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_global_figure(gr: dict, filename: str) -> None:
    """
    Two panels:
      Left: heatmap of |Pearson correlation| between parameters and metrics
      Right: bar chart of mean |correlation| per parameter (importance ranking)
    """
    corr_abs = np.abs(gr['corr'])   # (n_params, n_metrics)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Global Sensitivity — N={gr['corr'].shape[0]} random draws\n"
                 f"|Pearson correlation| between parameters and metrics", fontsize=11)

    # Heatmap
    im = ax1.imshow(corr_abs, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.8)
    ax1.set_xticks(range(len(gr['metric_names'])))
    ax1.set_xticklabels(gr['metric_names'], fontsize=8, rotation=30, ha='right')
    ax1.set_yticks(range(len(gr['param_names'])))
    ax1.set_yticklabels(gr['param_names'], fontsize=8)
    for i in range(len(gr['param_names'])):
        for j in range(len(gr['metric_names'])):
            ax1.text(j, i, f"{corr_abs[i,j]:.2f}", ha='center', va='center',
                     fontsize=7, color='black' if corr_abs[i,j] < 0.5 else 'white')
    plt.colorbar(im, ax=ax1, label='|correlation|')
    ax1.set_title('|Correlation| heatmap', fontsize=10)

    # Importance bar chart (mean |corr| per parameter, sorted)
    importance = corr_abs.mean(axis=1)
    order = np.argsort(importance)[::-1]
    pnames_sorted = [gr['param_names'][i] for i in order]
    imp_sorted    = importance[order]
    bars = ax2.barh(range(len(pnames_sorted)), imp_sorted, color='steelblue', alpha=0.8)
    ax2.set_yticks(range(len(pnames_sorted)))
    ax2.set_yticklabels(pnames_sorted, fontsize=9)
    ax2.set_xlabel('Mean |correlation| across all metrics', fontsize=9)
    ax2.set_title('Parameter importance ranking', fontsize=10)
    ax2.set_xlim(0, max(imp_sorted) * 1.25)
    for bar, val in zip(bars, imp_sorted):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va='center', fontsize=8)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_disc_figure(disc_results: dict, filename: str) -> None:
    """
    Side-by-side invest heatmaps: 11 bins vs 21 bins for 5 key sweeps.
    Rows = sweeps, columns = [11-bin invest, 21-bin invest].
    """
    names = list(disc_results.keys())
    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.5 * n))
    if n == 1:
        axes = axes[None, :]
    fig.suptitle('Discretization Robustness: 11-bin ($20k) vs 21-bin ($10k) wealth', fontsize=12)

    for row, name in enumerate(names):
        dr = disc_results[name]
        sd = dr['sweep_def']
        vals = [r['param_value'] for r in dr['run_11']]

        for col, (run_list, nbins, label) in enumerate([
            (dr['run_11'], 11, '11 bins ($20k)'),
            (dr['run_21'], 21, '21 bins ($10k)'),
        ]):
            ax = axes[row, col]
            inv_mat = np.array([r['pslice']['invest'] for r in run_list]).T  # (NW, n_vals)
            WU = run_list[0]['mdp'].wealth_unit
            wu_k = int(WU / 1000)
            ylabels = [f"${w*wu_k}k" for w in range(nbins)]
            im = ax.imshow(inv_mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=2, origin='lower')
            ax.set_title(f"{sd['label']} — {label}", fontsize=8)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels([f"{v:.3g}" for v in vals], fontsize=6, rotation=30)
            ax.set_yticks(range(nbins))
            ax.set_yticklabels(ylabels, fontsize=5)
            ax.set_xlabel(sd['label'], fontsize=7)
            ax.set_ylabel('Wealth ($)', fontsize=7)
            plt.colorbar(im, ax=ax, ticks=[0, 1, 2], label='invest', fraction=0.046)

    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    total_t0 = time.time()
    print('=== Comprehensive Sensitivity Analysis ===')
    print(f'Reference state: {S0}\n')

    # ---- Tier 1: Single-parameter sweeps ----
    print('TIER 1 — Reward weight sweeps (8 parameters)...')
    t0 = time.time()
    tier1_runs = {}
    for sweep_def in TIER1_SWEEPS:
        print(f"  Sweeping {sweep_def['label']} ({len(sweep_def['values'])} values)...")
        tier1_runs[sweep_def['name']] = run_1d_sweep(sweep_def)
    print(f"  Tier 1 1D done in {time.time()-t0:.1f}s\n")

    print('--- Tier 1 Policy Tables ---')
    for sweep_def in TIER1_SWEEPS:
        print_1d_table(sweep_def, tier1_runs[sweep_def['name']])

    print('\nGenerating sensitivity_1d.png...')
    plot_1d_figure(TIER1_SWEEPS, [tier1_runs[s['name']] for s in TIER1_SWEEPS],
                   'sensitivity_1d.png')

    # ---- Tier 1: 2D interaction grids ----
    print(f'\nTIER 1 — 2D interaction grids (4 grids x 25 MDPs)...')
    t0 = time.time()
    tier1_grids = []
    for gd in TIER1_GRIDS:
        print(f"  Grid: {gd['title']}...")
        tier1_grids.append(run_2d_sweep(gd))
    print(f"  Tier 1 2D done in {time.time()-t0:.1f}s\n")

    print('Generating sensitivity_2d.png...')
    plot_2d_figure(tier1_grids, 'sensitivity_2d.png')

    # ---- Tier 2: Dynamics ----
    print(f'\nTIER 2 — Transition dynamics ({sum(len(s["values"]) for s in TIER2_DYN)} MDPs)...')
    t0 = time.time()
    tier2_runs = {}
    for sweep_def in TIER2_DYN:
        print(f"  Sweeping {sweep_def['label']} ({len(sweep_def['values'])} values)...")
        tier2_runs[sweep_def['name']] = run_1d_sweep(sweep_def)
    print(f"  Tier 2 done in {time.time()-t0:.1f}s\n")

    print('--- Tier 2 Policy Tables ---')
    for sweep_def in TIER2_DYN:
        print_1d_table(sweep_def, tier2_runs[sweep_def['name']])

    print('\nGenerating sensitivity_dynamics.png...')
    plot_1d_figure(TIER2_DYN, [tier2_runs[s['name']] for s in TIER2_DYN],
                   'sensitivity_dynamics.png')

    # ---- Tier 3: Economics ----
    total_econ_mdps = sum(len(g['p1_values']) * len(g['p2_values']) for g in TIER3_ECON)
    print(f'\nTIER 3 — Economics grids ({total_econ_mdps} MDPs)...')
    t0 = time.time()
    tier3_grids = []
    for gd in TIER3_ECON:
        print(f"  Grid: {gd['title']}...")
        tier3_grids.append(run_2d_sweep(gd))
    print(f"  Tier 3 done in {time.time()-t0:.1f}s\n")

    print('Generating sensitivity_economics.png...')
    plot_2d_figure(tier3_grids, 'sensitivity_economics.png')

    # ---- Global sensitivity ----
    N_GLOBAL = 150
    print(f'\nGLOBAL SENSITIVITY — N={N_GLOBAL} random parameter draws...')
    t0 = time.time()
    global_result = run_global_sensitivity(N=N_GLOBAL)
    print(f"  Global done in {time.time()-t0:.1f}s\n")

    print('Generating sensitivity_global.png...')
    plot_global_figure(global_result, 'sensitivity_global.png')

    # Print top importance ranking
    corr_abs  = np.abs(global_result['corr'])
    importance = corr_abs.mean(axis=1)
    order     = np.argsort(importance)[::-1]
    print('\n  Parameter importance (mean |correlation|):')
    for i in order:
        print(f"    {global_result['param_names'][i]:>22}: {importance[i]:.3f}")

    # ---- Discretization check ----
    print(f'\nDISCRETIZATION CHECK — 5 sweeps at 11 vs 21 bins...')
    t0 = time.time()
    disc_result = run_disc_check()
    print(f"  Disc check done in {time.time()-t0:.1f}s\n")

    print('Generating sensitivity_disc.png...')
    plot_disc_figure(disc_result, 'sensitivity_disc.png')

    # ---- Summary ----
    total_elapsed = time.time() - total_t0
    print(f'\n=== All done in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min) ===')
    print('Output files:')
    for fn in ['sensitivity_1d.png', 'sensitivity_2d.png', 'sensitivity_dynamics.png',
               'sensitivity_economics.png', 'sensitivity_global.png', 'sensitivity_disc.png']:
        print(f'  {fn}')
