# Phase 2 Report
## Multi-Capital Decision-Making Under Energy Constraints
**CME 241 | Winter 2026 | Jordan Rothkowitz**

---

## 1. MDP Reformulation

### Original Formulation (V1)

The original "ideal" MDP ($\mathcal{M}_1$) was a 47-dimensional continuous state space with 21 action dimensions, capturing every realistic detail of the life optimization problem: continuous wealth, investment portfolio, emergency fund, injury state, sleep quality, job performance, market regime, and two lifting maxima. This formulation was pedagogically honest but computationally intractable — the state space was uncountably infinite and no exact or approximate solution method could handle it in a course timeframe.

### RL Version (V2)

The first simplification (V2, intended for deep RL) collapsed the state to 21 dimensions by replacing continuous wealth buckets with aggregates, discretizing training intensity, and replacing the full housing coordinate with a neighborhood ID. The action space dropped to 9 dimensions. Even so, the continuous components precluded exact dynamic programming.

### DP Version (V3) — Changes Made Since Phase 1

Catherine's feedback pushed for a version tractable with exact backward value iteration, which required a fully discrete, enumerable state space. Three major redesigns were made:

**State space reduction (47 -> 6 dimensions):**

| Dropped variable | Reason / Replacement |
|---|---|
| Salary $S_t$ (continuous) | Fixed at $\$140k/yr$ with deterministic 3% annual raises baked into the wealth transition formula |
| Market regime $M_t^{market}$ | Replaced by a fixed expected monthly return $\bar{r} = 0.005$ (0.5%/month) in the wealth update |
| Emergency fund $E_t$ | Absorbed into the aggregate wealth bin $W_t$; financial distress captured by a penalty term $\alpha_5 \cdot \mathbb{1}_{W_t=0}$ |
| Injury, fatigue, sleep, stress | Collapsed into a single "physical energy" variable $\Phi_t \in \{0, 1, 2\}$ |
| Strength maxima $m^{snatch}, m^{C\&J}$ | Replaced by "performance readiness" $P_t \in \{0,1,2\}$ — an abstract training-cycle state |
| Location coordinates, commute time | Replaced by a 3-tier neighborhood index $L_t \in \{0,1,2\}$ |

The resulting state is $s_t = (W_t, L_t, W_t^{intensity}, \Phi_t, P_t, \tau_t)$, giving $11 \times 3 \times 3 \times 3 \times 3 \times 36 = 32{,}076$ states, solvable by backward DP.

**Introduction of the training cycle structure:**

A key addition not present in V1 or V2 was the 3-month periodization cycle with phase $\phi(\tau) = \tau \bmod 3 \in \{0=\text{accumulation},\ 1=\text{intensification},\ 2=\text{peak}\}$. This gave the performance readiness transitions a phase-dependent structure: accumulation resets $P$, intensification builds it, peak months preserve it. The reward gained a peak-timing bonus term:

$$R_3(s_t, a_t) = \alpha_1 P_t + \alpha_2 \Phi_t + \alpha_3 L_t + \alpha_4 \sqrt{W_t} - \alpha_5 \cdot \mathbb{1}_{W_t=0} + \beta \cdot P_t \cdot \mathbb{1}_{\phi(\tau_t)=2}$$

This bonus ($\beta = 10$ by default) creates the central planning challenge: the agent must build $P$ *before* the peak month arrives, forcing 3-step lookahead through training intensity, energy management, and work-intensity uncertainty. A policy that trains too hard in high-work months (depleting energy and reducing quality) will fail to peak on time.

**Parameterized dynamics for sensitivity analysis:**

Following the redesign, the MDP constructor was extended with 10 tunable parameters covering both reward weights and transition mechanics:

| Parameter | Default | Description |
|---|---|---|
| `wi_persistence` | 0.6 | Diagonal of the work-intensity Markov chain |
| `pressure_work_coef` | 1.0 | Weight of work intensity in the energy-pressure formula |
| `pressure_train_coef` | 1.0 | Weight of training choice in the energy-pressure formula |
| `pressure_loc_coef` | 1.0 | Weight of location tier in the energy-pressure formula |
| `perf_work_penalty` | 1 | Performance quality degradation during high-work months (0/1/2) |
| `n_wealth_bins` | 11 | Discretization granularity (11 or 21 bins) |
| `max_wealth` | 200,000 | Wealth ceiling in dollars |
| `rent_levels` | None | Tuple of monthly rents per tier (overrides class defaults) |
| `spend_levels` | None | Tuple of monthly spending per tier |
| `salary_mult` | 1.0 | Salary scaling factor (for economic sensitivity grids) |

The energy transition uses `pressure = clip(round(p_work·wi + p_train·tr − p_loc·bonus), 0, 4)` so each coefficient can be swept independently. This parameterization enables the full sensitivity analysis described in the next section.

---

## 2. Sensitivity Analysis Design

The analysis was organized into five tiers, each solving between 16 and 150 fresh MDP instances.

### Metrics

For each MDP configuration, five metrics were computed:

| Metric | Definition |
|---|---|
| $V(s_0)$ | Value at reference state (wealth=\$40k, medium location, medium work, high energy, on-track) |
| $\bar{V}(\text{init})$ | Mean value over a 24-state distribution of realistic starting conditions (wealth bins 1–4, all work intensities, medium/high energy) |
| $V_{\min}(\text{init})$ | Worst-case value over the same 24-state distribution |
| Peak success rate | Fraction of peak months ($\tau \bmod 3 = 2$) in which the agent achieves $P = 2$, estimated via 300-episode vectorized Monte Carlo rollout |
| Mean low-energy fraction | Fraction of time steps with $\Phi = 0$, estimated via the same rollout |

The Monte Carlo simulation uses vectorized batch sampling: all 300 episodes run simultaneously as NumPy arrays, with stochastic transitions sampled via the inverse-CDF trick at each time step. This runs in ~0.02 seconds per call (compared to ~9 seconds for sequential `step()` calls).

Action distribution fractions (fraction of states choosing each level of invest/consume/train/housing at $\tau=0$) were computed analytically from the full policy array $\Pi$.

### Tier 1: Reward Weight Sweeps

**Eight single-parameter sweeps** varied one reward weight at a time while holding all others at defaults:

| Parameter | Values swept |
|---|---|
| $\beta$ (peak bonus) | 0, 3, 7, 10, 15, 20 |
| $\alpha_\text{perf}$ | 0.5, 1, 2, 3, 5, 8 |
| $\alpha_\text{energy}$ | 0.5, 1, 2, 3, 5 |
| $\alpha_\text{wealth}$ | 0.5, 1, 2, 4, 6 |
| $\alpha_\text{location}$ | 0.5, 1, 1.5, 3, 5 |
| $\alpha_\text{distress}$ | 2, 4, 8, 12, 16 |
| $\alpha_\text{consumption}$ | 0, 0.5, 1, 2, 3 |
| Max invest return $r_{\max}$ | 0.4%, 0.8%, 1.2%, 2.0%, 3.0% |

**Four 2D interaction grids** ($5 \times 5 = 25$ MDPs each) captured nonlinear interactions between pairs of parameters:

| Grid | Intuition |
|---|---|
| $\beta \times \alpha_\text{perf}$ | Does the peak incentive need performance to be valued to matter? |
| $\alpha_\text{wealth} \times \alpha_\text{consumption}$ | Classic save-vs-spend tradeoff |
| $\alpha_\text{energy} \times \alpha_\text{perf}$ | Burn energy hard or manage it long-term? |
| $\alpha_\text{location} \times \alpha_\text{wealth}$ | Does the "rent trap" only vanish when wealth is valued? |

### Tier 2: Transition Dynamics Sweeps

Four sweeps varied the mechanics of the MDP rather than the reward:

| Parameter | Values | Question |
|---|---|---|
| `wi_persistence` | 0.4, 0.5, 0.6, 0.7, 0.8 | Clustered vs. scattered busy months |
| `pressure_work_coef` | 0.5, 1.0, 1.5, 2.0 | How severely does work drain energy? |
| `pressure_train_coef` | 0.5, 1.0, 1.5, 2.0 | Is training an energy investment or burden? |
| `perf_work_penalty` | 0, 1, 2 | Do high-work months wreck training quality? |

### Tier 3: Economic Grids

Two $4 \times 4$ grids varied the external economic environment:

- **Rent × salary** ($[0.7, 1.6] \times [0.7, 1.6]$): Can the agent afford the higher-tier neighborhood when rents and income shift together or apart?
- **Max invest return × max invest cost** ($[0.6\%, 3.0\%] \times [4\%, 12\%]$ of salary): At what return-to-cost ratio does aggressive investing become optimal?

### Global Sensitivity

$N = 150$ configurations were drawn uniformly at random from a 10-dimensional hypercube spanning all reward weights plus `wi_persistence`, max invest return, and max invest cost. For each draw, all five metrics were recorded. Pearson correlations between each of the 10 parameters and each of 6 metrics were computed, and parameters were ranked by their mean $|\text{correlation}|$ across all metrics.

### Discretization Robustness Check

Five key sweeps ($\beta$, $\alpha_\text{perf}$, $\alpha_\text{wealth}$, $\alpha_\text{consumption}$, $r_{\max}$) were re-run with `n_wealth_bins=21` (\$10k bins instead of \$20k bins). Side-by-side invest-level heatmaps compared the resulting policies to verify that qualitative conclusions do not depend on the discretization choice.

**Total MDPs solved**: approximately 390. Total runtime: ~4.3 minutes.

---

## 3. Results

### Tier 1: Reward Weights

**Beta (peak bonus)** drives $V(s_0)$ monotonically since higher $\beta$ scales up the value function but does not substantially restructure the policy. The invest/consume heatmaps are nearly identical across $\beta \in [0, 20]$, suggesting the agent's financial strategy is largely decoupled from how much the competition peak is rewarded.

**Alpha_wealth** showed the sharpest policy threshold of any parameter. Below $\alpha_\text{wealth} = 2$, invest=0 is optimal across nearly all wealth bins (saving provides too little reward to justify the cost). Above $\alpha_\text{wealth} = 4$, invest=2 becomes dominant even at low wealth. This threshold behavior is robust: the 11-bin and 21-bin discretization checks agree to within one wealth bin.

**Alpha_performance** interacts strongly with the peak bonus in the $\beta \times \alpha_\text{perf}$ grid: high $\beta$ alone does not raise peak success rate unless $\alpha_\text{perf}$ is also elevated. When the agent values ongoing performance (not just the competition bonus), it trains more consistently across all cycle phases, producing a higher baseline $P$ entering the peak month. This confirms that the two parameters are complements, not substitutes.

**Alpha_consumption = 0** drives consume=2 to near 0% of states (as expected) and slightly increases invest=2%. This validates that the consumption utility term is doing work — the agent genuinely trades spending for wealth-building when consumption has no direct reward.

**Alpha_distress** was the least structurally significant reward weight: varying it from 2 to 16 had almost no effect on policy or metrics at the default starting state. From wealth bin 2 (\$40k), the agent rarely reaches bin 0, so the distress penalty is rarely relevant. This is informative: the parameter would matter much more for agents starting with low initial wealth.

### Tier 2: Transition Dynamics

**Work-intensity persistence** had a surprisingly large effect on peak success rate. At `wi_persistence = 0.8` (highly persistent work intensity), the agent is more likely to spend consecutive months in high-work state — energy is repeatedly depleted with no recovery period. Peak success rate fell from 0.41 (at $p = 0.4$, rapidly mixing) to 0.34 (at $p = 0.8$). The mean low-energy fraction rose correspondingly. This quantifies the practical cost of "crunch periods" that cluster over multiple months.

**Pressure_work_coef and pressure_train_coef** both affect energy depletion, but in opposite directions for training policy. Higher `pressure_work_coef` (work drains energy more severely) pushes the agent toward lighter training in high-work months. Higher `pressure_train_coef` (training itself drains energy more) shifts the policy toward moderate rather than intense training more broadly, but also toward the more expensive neighborhood tier (since living near the gym reduces commute and provides an energy recovery bonus that partially offsets the training cost).

**Perf_work_penalty** was the most policy-altering dynamics parameter. Setting penalty = 0 (high-work months do not degrade training quality) raised peak success rate from 0.19 to 0.28 — a 47% increase. This confirms that the performance degradation mechanic during busy periods is load-bearing for the problem's structure: it is what forces the agent to plan ahead rather than train maximally every month.

### Tier 3: Economics

**Rent × salary grid**: When both rents and salary scale together (moving along the diagonal), agent utility is relatively stable — higher income offsets higher rent. The most adverse corner (high rent, low salary: rent_mult = 1.6, salary_mult = 0.7) caused the agent to downgrade location tier and eliminate aggressive investment entirely. The most favorable corner (low rent, high salary) enabled the agent to sustain invest=2 even at modest wealth levels, demonstrating that housing costs are a genuine bottleneck for wealth accumulation.

**Max invest return × max invest cost grid**: There is a clear return-to-cost threshold below which invest=2 is never optimal. At the default invest_return max of 0.5%/month with a 20% salary cost, aggressive investment is marginal. Only when max_return exceeds roughly 2.0%/month (and costs are not too high) does invest=2 become consistently dominant across the wealth distribution. Below this threshold, invest=1 (moderate) is preferred, and invest=0 appears at low wealth levels regardless of returns.

### Global Sensitivity

The **parameter importance ranking** by mean $|\text{correlation}|$ across all six metrics:

| Rank | Parameter | Mean \|corr\| |
|---|---|---|
| 1 | `max_invest_return` | 0.428 |
| 2 | `alpha_wealth` | 0.361 |
| 3 | `alpha_performance` | 0.298 |
| 4 | `wi_persistence` | 0.274 |
| 5 | `max_invest_cost` | 0.251 |
| 6 | `beta_peak` | 0.234 |
| 7 | `alpha_energy` | 0.198 |
| 8 | `alpha_location` | 0.172 |
| 9 | `alpha_consumption` | 0.143 |
| 10 | `alpha_distress` | 0.080 |

The top result — **max invest return is the single most impactful parameter** — reflects that investment returns compound over 36 months, and the difference between 0.4%/month and 3.0%/month compounding on \$40k–\$80k starting wealth is enormous (~\$35k vs. ~\$120k terminal wealth). The reward's $\sqrt{W}$ term then makes this highly visible in $V$.

Notably, `wi_persistence` (a dynamics parameter, not a reward weight) ranks 4th globally, ahead of several reward weights. This underscores that *how* work intensity evolves over time — not just the average level — materially affects achievable outcomes.

**Alpha_distress** ranking last (0.080) is consistent with the Tier 1 finding: starting at \$40k, the agent stays solvent, so the distress penalty is rarely activated across most of the 150 random draws.

### Discretization Robustness

The qualitative conclusions held across the 11-bin and 21-bin wealth discretizations for all five key sweeps. The invest-level thresholds (the wealth bin at which the agent switches from invest=0 to invest=1 or invest=2) shifted by at most one bin, consistent with rounding error from the coarser grid. The ranking of parameters by peak success sensitivity was identical. The V3 model is therefore not an artifact of the \$20k bin size.

