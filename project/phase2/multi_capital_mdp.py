"""
Multi-Capital Decision-Making Under Energy Constraints

MDP for optimizing life decisions across financial, energy, and human capital
for an early-career professional who is also a competitive weightlifter.

Implements V3: a fully discrete, 6-dimensional state space tractable with
exact backward value iteration (32,076 states with default 11 wealth bins).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable, Generic, TypeVar, Callable, Optional
from abc import ABC, abstractmethod
import numpy as np


# ============================================================================
# Minimal MDP Framework (subset of course materials)
# ============================================================================

S = TypeVar('S')
A = TypeVar('A')


class State(ABC, Generic[S]):
    state: S


@dataclass(frozen=True)
class Terminal(State[S]):
    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    state: S

    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return self.state < other.state


class Distribution(ABC, Generic[S]):
    @abstractmethod
    def sample(self) -> S:
        pass


class SampledDistribution(Distribution[S]):
    def __init__(self, sampler: Callable[[], S]):
        self.sampler = sampler

    def sample(self) -> S:
        return self.sampler()


class MarkovDecisionProcess(ABC, Generic[S, A]):
    @abstractmethod
    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        pass

    @abstractmethod
    def step(
        self,
        state: NonTerminal[S],
        action: A
    ) -> Distribution[Tuple[State[S], float]]:
        pass


# ============================================================================
# Version 3: Simplified DP Version
# ============================================================================

@dataclass(frozen=True)
class StateV3:
    """Fully discrete, 6-dimensional state.  |S| = 11x3x3x3x3x36 = 32,076 (default bins).

    wealth        {0,...,n_wealth_bins-1} : wealth in wealth_unit bins ($0-max_wealth)
    location      {0,1,2}   : neighborhood tier (cheap-far / medium / expensive-close)
    work_intensity{0,1,2}   : monthly work demand (low/medium/high) -- exogenous
    energy        {0,1,2}   : physical energy level (low/medium/high)
    performance   {0,1,2}   : training readiness (undertrained/on-track/peaked)
    time          {0,...,35}: month index (3-year horizon)
    """
    wealth: int
    location: int
    work_intensity: int
    energy: int
    performance: int
    time: int

    def __lt__(self, other: StateV3) -> bool:
        return (self.wealth, self.location, self.work_intensity,
                self.energy, self.performance, self.time) < \
               (other.wealth, other.location, other.work_intensity,
                other.energy, other.performance, other.time)


@dataclass(frozen=True)
class ActionV3:
    """Fully discrete, 4-dimensional action.  |A| = 3^4 = 81."""
    housing: int      # {-1,0,1}  : downgrade / stay / upgrade location tier
    invest: int       # {0,1,2}   : investment aggressiveness (low/medium/high)
    training: int     # {0,1,2}   : training intensity (light / moderate / intense)
    consumption: int  # {0,1,2}   : discretionary spend (frugal / moderate / generous)


class MultiCapitalEnergyMDP(MarkovDecisionProcess[StateV3, ActionV3]):
    """
    MDP for multi-capital optimization under energy constraints (V3).

    Default state space : 11 x 3 x 3 x 3 x 3 x 36 = 32,076 states
    Action space: 3^4 = 81 actions per state
    Solution    : exact backward value iteration (~2.6M Q-values)

    All reward weights, investment parameters, dynamics knobs, and structural
    parameters are instance attributes so they can be varied for sensitivity
    analysis without subclassing.
    """

    # --- Structural constants (class-level defaults; overridden per-instance) ---
    S_BASE  = 140_000 / 12   # base monthly salary (~$11,667), pre-raise
    R_RAISE = 1.03            # annual salary raise multiplier
    RENTS   = {0: 1_500, 1: 2_200, 2: 3_000}   # monthly rent by location tier
    SPEND   = {0: 1_000, 1: 2_000, 2: 3_500}   # monthly discretionary spend

    # --- Work intensity transition matrix (rows = current, cols = next) ---
    # This class-level default is replaced by self.work_intensity_matrix in __init__
    WORK_INTENSITY_MATRIX = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6],
    ])

    # --- Energy transition ---
    # Dictionary keys: current energy level
    # Values: 5x3 array with rows ordered by pressure=4,3,2,1,0 and cols ordered by next energy level 0,1,2
    # pressure = work_intensity + training - location_bonus; clipped to [0,4]
    # to get pressure=4: 2+2-0
    # to get pressure=3: 2+1-0, 1+2-0, 2+2-1,
    # to get pressure=2: 1+1-0, 2+0-0, 0+2-0, 2+1-1, 1+2-1, 2+2-2
    # to get pressure=1: 1+0-0, 0+1-0, 1+1-1, 0+2-1, 2+0-1, 1+2-2, 2+1-2
    # to get pressure=0: 0+0-0, 1+0-1, 0+1-1, 0+0-1, 0+1-2, 1+0-2, 0+0-2
    ENERGY_TABLE = {
        # Current energy = 0
        0: np.array([[0.80, 0.15, 0.05], # transition probabilities under pressure=4
                     [0.60, 0.35, 0.05], # transition probabilities under pressure=3
                     [0.40, 0.45, 0.15], # transition probabilities under pressure=2
                     [0.25, 0.50, 0.25], # transition probabilities under pressure=1
                     [0.15, 0.45, 0.40]]), # transition probabilities under pressure=0
        # Current energy = 1
        1: np.array([[0.70, 0.25, 0.05],
                     [0.45, 0.45, 0.10],
                     [0.20, 0.55, 0.25],
                     [0.10, 0.55, 0.35],
                     [0.05, 0.45, 0.50]]),
        # Current energy = 2
        2: np.array([[0.70, 0.25, 0.05],
                     [0.30, 0.50, 0.20],
                     [0.10, 0.50, 0.40],
                     [0.05, 0.35, 0.60],
                     [0.05, 0.20, 0.75]]),
    }

    # --- Performance readiness transition ---
    # Indexed by [phase][current_P][quality] -> np.array([P(0), P(1), P(2)])
    # quality = min(training, energy), reduced by perf_work_penalty if work_intensity == 2
    # phase = time % 3: 0=accumulation, 1=intensification, 2=peak month

    # Example: If we're undertrained (P=0) during accumulation (phase=0),
    # quality 0 -> no progress (we stay in P=0 with prob 1.0)
    # quality 1 -> 30% chance we move to P=1
    # quality 2 -> 50% chance we move to P=1
    # Note no possibility to jump straight to P=2 in accumulation phase.
    PERFORMANCE_TABLE = {
        # Outer key: training phase 0 (Accumulation) builds volume; P resets after peak
        0: {
            # Inner key: current performance level
            # Value matrix cols: next performance level 0,1,2
            0: [np.array([1.0, 0.0, 0.0]), # probabilities for next performance level under quality=0
                np.array([0.7, 0.3, 0.0]), # probabilities for next performance level under quality=1
                np.array([0.5, 0.5, 0.0])], # probabilities for next performance level under quality=2
            1: [np.array([0.3, 0.7, 0.0]),
                np.array([0.1, 0.9, 0.0]),
                np.array([0.0, 0.9, 0.1])],
            2: [np.array([0.3, 0.7, 0.0]),
                np.array([0.1, 0.9, 0.0]),
                np.array([0.0, 0.8, 0.2])],
        },
        # Training phase 1 (Intensification)
        1: {
            0: [np.array([1.0, 0.0, 0.0]),
                np.array([0.7, 0.3, 0.0]),
                np.array([0.3, 0.7, 0.0])],
            1: [np.array([0.2, 0.8, 0.0]),
                np.array([0.0, 0.8, 0.2]),
                np.array([0.0, 0.5, 0.5])],
            2: [np.array([0.0, 0.6, 0.4]),
                np.array([0.0, 0.4, 0.6]),
                np.array([0.0, 0.2, 0.8])],
        },
        # Training phase 2 (Peak)
        2: {
            0: [np.array([0.9, 0.1, 0.0]),
                np.array([0.7, 0.3, 0.0]),
                np.array([0.5, 0.5, 0.0])],
            1: [np.array([0.2, 0.7, 0.1]),
                np.array([0.0, 0.85, 0.15]),
                np.array([0.0, 0.5,  0.5])],
            2: [np.array([0.1, 0.3, 0.6]),
                np.array([0.0, 0.15, 0.85]),
                np.array([0.0, 0.4,  0.6])],
        },
    }

    def __init__(
        self,
        gamma: float = 0.99,          # discount factor
        time_horizon: int = 36,       # 3 years of monthly decisions
        random_seed: int = 42,
        # Reward weights
        alpha_performance: float = 3.0,
        alpha_energy: float = 2.0,
        alpha_location: float = 1.5,
        alpha_wealth: float = 2.0,
        alpha_distress: float = 8.0,
        alpha_consumption: float = 1.0,
        beta_peak: float = 10.0,
        # Investment parameters: effective monthly return per investment level [0, 1, 2]
        # and opportunity cost per invest level (aggressive investment reduces disposable income)
        invest_returns: Tuple[float, float, float] = (0.002, 0.006, 0.012),
        invest_costs: Tuple[float, float, float] = (0.00, 0.04, 0.09),
        # Dynamics knobs (for sensitivity analysis over model structure)
        wi_persistence: float = 0.6,          # diagonal of work-intensity Markov chain [0,1]
        pressure_work_coef: float = 1.0,      # weight on work_intensity in pressure formula
        pressure_train_coef: float = 1.0,     # weight on training in pressure formula
        pressure_loc_coef: float = 1.0,       # weight on location_bonus in pressure formula
        perf_work_penalty: int = 1,           # quality reduction when wi==2 (0=none, 1=default, 2=harsh)
        # Discretization / scale parameters
        n_wealth_bins: int = 11,              # number of wealth bins (11 → $0–$200k in $20k steps)
        max_wealth: int = 200_000,            # maximum wealth in dollars
        # Optional overrides for rent and spend levels (tuples of 3 floats, one per tier)
        rent_levels: Optional[Tuple[float, float, float]] = None,
        spend_levels: Optional[Tuple[float, float, float]] = None,
        # Salary scaling (for economics sensitivity analysis)
        salary_mult: float = 1.0,
    ):
        self.gamma        = gamma
        self.time_horizon = time_horizon
        self.rng          = np.random.RandomState(random_seed)

        # Reward weights
        self.alpha = {
            'performance': alpha_performance,
            'energy':      alpha_energy,
            'location':    alpha_location,
            'wealth':      alpha_wealth,
            'distress':    alpha_distress,
            'consumption': alpha_consumption,
        }
        self.beta_peak = beta_peak

        # Investment economics
        self.invest_returns = invest_returns
        self.invest_costs   = invest_costs

        # Dynamics knobs
        self.pressure_work_coef  = pressure_work_coef
        self.pressure_train_coef = pressure_train_coef
        self.pressure_loc_coef   = pressure_loc_coef
        self.perf_work_penalty   = perf_work_penalty

        # Work-intensity Markov chain generated from persistence parameter
        # Rows = current state, columns = next state; symmetric off-diagonals
        p   = float(wi_persistence)
        off = (1.0 - p) / 2.0
        self.work_intensity_matrix = np.array([
            [p,   off, off],
            [off, p,   off],
            [off, off, p  ],
        ])

        # Wealth discretization
        self.n_wealth_bins = n_wealth_bins
        self.wealth_unit   = max_wealth / (n_wealth_bins - 1)  # e.g. $20k for 11 bins

        # Rent and spend levels (instance-level, default to class constants)
        if rent_levels is not None:
            self._rents = tuple(rent_levels)
        else:
            self._rents = (self.RENTS[0], self.RENTS[1], self.RENTS[2])

        if spend_levels is not None:
            self._spend = tuple(spend_levels)
        else:
            self._spend = (self.SPEND[0], self.SPEND[1], self.SPEND[2])

        # Effective base salary (scales with salary_mult for economics analysis)
        self._salary_base = self.S_BASE * salary_mult

    # ------------------------------------------------------------------
    # MDP interface
    # ------------------------------------------------------------------

    def actions(self, state: NonTerminal[StateV3]) -> Iterable[ActionV3]:
        """All valid actions; housing moves constrained by current location tier."""
        s = state.state
        housing_choices = []
        if s.location > 0:
            housing_choices.append(-1)
        housing_choices.append(0)
        if s.location < 2:
            housing_choices.append(1)
        return [
            ActionV3(h, i, tr, c)
            for h in housing_choices
            for i in [0, 1, 2]
            for tr in [0, 1, 2]
            for c in [0, 1, 2]
        ]

    def _wealth_delta(self, w: int, new_location: int, invest: int, consumption: int, t: int) -> float:
        """Monthly change in wealth (dollars) given deterministic state components."""
        salary = self._salary_base * (self.R_RAISE ** (t // 12))
        return (
            salary
            - self._rents[new_location]
            - self._spend[consumption]
            - self.invest_costs[invest] * self._salary_base
            + self.invest_returns[invest] * w * self.wealth_unit
        )

    def enumerate_transitions(
        self,
        state: NonTerminal[StateV3],
        action: ActionV3
    ) -> list:
        """
        Return all reachable (next_state, probability, reward) triples.
        Used for exact backward value iteration.
        """
        s = state.state
        reward = self._compute_reward(s, action)

        if s.time >= self.time_horizon - 1:
            return [(Terminal(s), 1.0, reward + self._terminal_reward(s))]

        new_location = int(np.clip(s.location + action.housing, 0, 2))
        new_time = s.time + 1
        delta = self._wealth_delta(s.wealth, new_location, action.invest, action.consumption, s.time)
        new_wealth = int(np.clip(round(s.wealth + delta / self.wealth_unit), 0, self.n_wealth_bins - 1))

        wi_probs   = self.work_intensity_matrix[s.work_intensity]
        en_probs   = self._energy_probs(s.energy, s.work_intensity, action.training, new_location)
        phase      = s.time % 3
        perf_probs = self._performance_probs(
            s.performance, action.training, s.energy, s.work_intensity, phase
        )

        transitions = []
        for wi in range(3):
            for en in range(3):
                for perf in range(3):
                    prob = wi_probs[wi] * en_probs[en] * perf_probs[perf]
                    if prob < 1e-12:
                        continue
                    ns = StateV3(
                        wealth=new_wealth, location=new_location,
                        work_intensity=wi, energy=en, performance=perf, time=new_time,
                    )
                    transitions.append((NonTerminal(ns), prob, reward))
        return transitions

    def step(
        self,
        state: NonTerminal[StateV3],
        action: ActionV3
    ) -> Distribution[Tuple[State[StateV3], float]]:
        """Return a SampledDistribution over (next_state, reward). Used for RL simulation."""
        s = state.state

        def sample_transition() -> Tuple[State[StateV3], float]:
            reward = self._compute_reward(s, action)
            if s.time >= self.time_horizon - 1:
                return Terminal(s), reward + self._terminal_reward(s)

            new_location = int(np.clip(s.location + action.housing, 0, 2))
            new_time = s.time + 1
            delta = self._wealth_delta(s.wealth, new_location, action.invest, action.consumption, s.time)
            new_wealth = int(np.clip(round(s.wealth + delta / self.wealth_unit), 0, self.n_wealth_bins - 1))

            new_work_intensity = self.rng.choice(
                [0, 1, 2], p=self.work_intensity_matrix[s.work_intensity]
            )
            new_energy = self.rng.choice(
                [0, 1, 2],
                p=self._energy_probs(s.energy, s.work_intensity, action.training, new_location)
            )
            phase = s.time % 3
            new_performance = self.rng.choice(
                [0, 1, 2],
                p=self._performance_probs(
                    s.performance, action.training, s.energy, s.work_intensity, phase
                )
            )
            next_s = StateV3(
                wealth=new_wealth, location=new_location,
                work_intensity=new_work_intensity, energy=new_energy,
                performance=new_performance, time=new_time,
            )
            return NonTerminal(next_s), reward

        return SampledDistribution(sample_transition)

    # ------------------------------------------------------------------
    # Transition probability helpers
    # ------------------------------------------------------------------

    def _energy_probs(
        self, current: int, work_intensity: int, training: int, location: int
    ) -> np.ndarray:
        """
        Energy depleted by work+training pressure; reduced by short commute (location=2).
        pressure = pressure_work_coef*wi + pressure_train_coef*tr - pressure_loc_coef*bonus
        ENERGY_TABLE[current] rows are ordered pressure=4,3,2,1,0 (row 0 = highest pressure).
        """
        location_bonus = 1 if location == 2 else 0
        pressure = int(np.clip(round(
            self.pressure_work_coef  * work_intensity
            + self.pressure_train_coef * training
            - self.pressure_loc_coef  * location_bonus
        ), 0, 4))
        row = 4 - pressure
        probs = self.ENERGY_TABLE[current][row].copy()
        return probs / probs.sum()

    def _performance_probs(
        self,
        current: int,
        training: int,
        energy: int,
        work_intensity: int,
        phase: int,
    ) -> np.ndarray:
        """
        Phase-aware performance readiness transition.
        quality = min(training, energy), degraded by perf_work_penalty if work_intensity == 2.
        """
        quality = min(training, energy)
        if work_intensity == 2:
            quality = max(0, quality - self.perf_work_penalty)
        return self.PERFORMANCE_TABLE[phase][current][quality]

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, state: StateV3, action: ActionV3) -> float:
        """
        R_3(s, a) = alpha1*P_t + alpha2*Phi_t + alpha3*L_t + alpha4*sqrt(W_t)
                    - alpha5*1[W_t=0] + alpha6*a_consumption
                    + beta*P_t*1[phase(tau_t)=2]
        """
        a = self.alpha
        phase = state.time % 3
        return float(
            a['performance'] * state.performance
            + a['energy']      * state.energy
            + a['location']    * state.location
            + a['wealth']      * np.sqrt(state.wealth)
            - a['distress']    * (1 if state.wealth == 0 else 0)
            + a['consumption'] * action.consumption
            + self.beta_peak   * state.performance * (1 if phase == 2 else 0)
        )

    def _terminal_reward(self, state: StateV3) -> float:
        """R_T = 10*W_T + 5*P_T"""
        return 10.0 * state.wealth + 5.0 * state.performance


# ============================================================================
# Solvers
# ============================================================================

def solve_mdp(mdp: MultiCapitalEnergyMDP) -> Tuple[dict, dict]:
    """
    Backward value iteration (dict-based). Correct but slow (~4 min).
    Use solve_mdp_fast for sensitivity analysis.
    """
    NW = mdp.n_wealth_bins
    all_states = [
        StateV3(w, l, wi, en, perf, t)
        for w    in range(NW)
        for l    in range(3)
        for wi   in range(3)
        for en   in range(3)
        for perf in range(3)
        for t    in range(mdp.time_horizon)
    ]

    V  = {}
    pi = {}

    for t in reversed(range(mdp.time_horizon)):
        for s in (s for s in all_states if s.time == t):
            nt = NonTerminal(s)
            best_val    = -np.inf
            best_action = None

            for action in mdp.actions(nt):
                q = 0.0
                for ns, prob, reward in mdp.enumerate_transitions(nt, action):
                    if isinstance(ns, Terminal):
                        q += prob * reward
                    else:
                        q += prob * (reward + mdp.gamma * V.get(ns.state, 0.0))
                if q > best_val:
                    best_val    = q
                    best_action = action

            V[s]  = best_val
            pi[s] = best_action

    return V, pi


def solve_mdp_fast(mdp: MultiCapitalEnergyMDP) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized backward value iteration using numpy arrays.

    Returns:
        V  : float array, shape (NW, 3, 3, 3, 3, T)  -- V[w,l,wi,en,perf,t]
        PI : int8 array,  shape (NW, 3, 3, 3, 3, T, 4) -- PI[...,t,:] = (h,inv,tr,c)
             housing stored as h+1 (0/1/2 = downgrade/stay/upgrade)
    where NW = mdp.n_wealth_bins (default 11).
    """
    T   = mdp.time_horizon
    NW  = mdp.n_wealth_bins
    WU  = mdp.wealth_unit
    wi_mat = mdp.work_intensity_matrix   # (3, 3)

    # Precompute energy table: en_table[en, wi, tr, nl, en'] shape (3,3,3,3,3)
    en_table = np.zeros((3, 3, 3, 3, 3))
    for en in range(3):
        for wi in range(3):
            for tr in range(3):
                for nl in range(3):
                    en_table[en, wi, tr, nl] = mdp._energy_probs(en, wi, tr, nl)

    # Precompute performance table: perf_table[ph, perf, tr, en, wi, perf'] (3,3,3,3,3,3)
    perf_table = np.zeros((3, 3, 3, 3, 3, 3))
    for ph in range(3):
        for perf in range(3):
            for tr in range(3):
                for en in range(3):
                    for wi in range(3):
                        perf_table[ph, perf, tr, en, wi] = mdp._performance_probs(
                            perf, tr, en, wi, ph
                        )

    # Value and policy arrays
    V_all  = np.zeros((NW, 3, 3, 3, 3, T))
    PI_all = np.zeros((NW, 3, 3, 3, 3, T, 4), dtype=np.int8)

    # V_next holds V[w,l,wi,en,perf] for time t+1
    V_next = np.zeros((NW, 3, 3, 3, 3))

    # Index grids for vectorised reward computation
    W_idx    = np.arange(NW)
    L_idx    = np.arange(3)
    WI_idx   = np.arange(3)
    EN_idx   = np.arange(3)
    PERF_idx = np.arange(3)

    for t in reversed(range(T)):
        phase = t % 3

        # Base reward (state-only) — shape (NW,3,3,3,3)
        W5    = W_idx[:, None, None, None, None]
        L5    = L_idx[None, :, None, None, None]
        EN5   = EN_idx[None, None, None, :, None]
        PERF5 = PERF_idx[None, None, None, None, :]
        a = mdp.alpha

        R_base = (
            a['performance'] * PERF5
            + a['energy']    * EN5
            + a['location']  * L5
            + a['wealth']    * np.sqrt(W5.astype(float))
            - a['distress']  * (W5 == 0)
            + mdp.beta_peak  * PERF5 * (1 if phase == 2 else 0)
        )
        R_base = np.broadcast_to(R_base, (NW, 3, 3, 3, 3)).copy()

        Q_best = np.full((NW, 3, 3, 3, 3), -np.inf)
        PI_t   = np.zeros((NW, 3, 3, 3, 3, 4), dtype=np.int8)

        for h_raw in [-1, 0, 1]:
            # New location for each current l: shape (3,)
            nl = np.clip(L_idx + h_raw, 0, 2)  # (3,)

            for inv in range(3):
                for tr in range(3):
                    for c in range(3):
                        # Immediate reward (adds consumption utility)
                        R_act = R_base + a['consumption'] * c

                        # New wealth: shape (NW, 3) -- varies with w and l
                        salary = mdp._salary_base * (mdp.R_RAISE ** (t // 12))
                        rents  = np.array([mdp._rents[nl[l]] for l in range(3)], dtype=float)
                        delta  = (
                            salary
                            - rents[None, :]
                            - mdp._spend[c]
                            - mdp.invest_costs[inv]   * mdp._salary_base
                            + mdp.invest_returns[inv] * W_idx[:, None] * WU
                        )  # shape (NW, 3)
                        nw = np.clip(
                            np.round(W_idx[:, None] + delta / WU).astype(int), 0, NW - 1
                        )  # shape (NW, 3)

                        if t == T - 1:
                            # Terminal: add terminal reward, no future value
                            R_term = 10.0 * W5 + 5.0 * PERF5   # (NW,1,1,1,3) broadcast
                            Q = R_act + np.broadcast_to(R_term, (NW, 3, 3, 3, 3))
                        else:
                            # Gather V_next at (nw[w,l], nl[l]) using full 5-index advanced indexing
                            _wi = np.arange(3)
                            _en = np.arange(3)
                            _pf = np.arange(3)
                            V_future = V_next[
                                nw[:, :, None, None, None],             # (NW, 3, 1, 1, 1)
                                nl[None, :, None, None, None],          # (1,  3, 1, 1, 1)
                                _wi[None, None, :, None, None],         # (1,  1, 3, 1, 1)
                                _en[None, None, None, :, None],         # (1,  1, 1, 3, 1)
                                _pf[None, None, None, None, :],         # (1,  1, 1, 1, 3)
                            ]  # shape (NW, 3, 3, 3, 3) = (w, l, wi', en', perf')

                            # Step 1: contract over wi' using wi_mat[wi, wi']
                            V1 = np.einsum('bA,wlAep->wlbep', wi_mat, V_future)

                            # Step 2: contract over en' using en_table[en,wi,tr,nl[l],en']
                            en_p = en_table[:, :, tr, nl, :]   # (en, wi, l, en') = (3,3,3,3)
                            en_p = en_p.transpose(2, 0, 1, 3)  # (l, en, wi, en')
                            V2 = np.einsum('lebN,wlbNp->wlbep', en_p, V1)

                            # Step 3: contract over perf' using perf_table[phase,perf,tr,en,wi,perf']
                            perf_p = perf_table[phase, :, tr, :, :, :]  # (perf, en, wi, perf')
                            V3 = np.einsum('Pebp,wlbep->wlbeP', perf_p, V2)

                            Q = R_act + mdp.gamma * V3

                        # Mask invalid housing moves
                        if h_raw == -1:
                            Q[:, 0, :, :, :] = -np.inf   # can't downgrade from l=0
                        elif h_raw == 1:
                            Q[:, 2, :, :, :] = -np.inf   # can't upgrade from l=2

                        improved = Q > Q_best
                        Q_best = np.where(improved, Q, Q_best)
                        PI_t[..., 0] = np.where(improved, h_raw + 1, PI_t[..., 0])  # store as 0/1/2
                        PI_t[..., 1] = np.where(improved, inv,       PI_t[..., 1])
                        PI_t[..., 2] = np.where(improved, tr,        PI_t[..., 2])
                        PI_t[..., 3] = np.where(improved, c,         PI_t[..., 3])

        V_next          = Q_best
        V_all[..., t]   = Q_best
        PI_all[..., t, :] = PI_t

    return V_all, PI_all


def policy_at(PI: np.ndarray, s: StateV3) -> ActionV3:
    """Extract optimal action for a given state from the fast-solver policy array."""
    raw = PI[s.wealth, s.location, s.work_intensity, s.energy, s.performance, s.time]
    return ActionV3(
        housing     = int(raw[0]) - 1,   # stored as h+1
        invest      = int(raw[1]),
        training    = int(raw[2]),
        consumption = int(raw[3]),
    )


def value_at(V: np.ndarray, s: StateV3) -> float:
    """Extract optimal value for a given state from the fast-solver value array."""
    return float(V[s.wealth, s.location, s.work_intensity, s.energy, s.performance, s.time])


# ============================================================================
# Example usage
# ============================================================================

def example_usage():
    import time

    mdp = MultiCapitalEnergyMDP(gamma=0.99, time_horizon=36, random_seed=42)
    wu_k = int(mdp.wealth_unit / 1000)  # wealth unit in $k

    # --- Single-step sample (sanity check) ---
    s0 = StateV3(wealth=2, location=1, work_intensity=1, energy=2, performance=1, time=0)
    initial_state = NonTerminal(s0)

    actions = list(mdp.actions(initial_state))
    print(f"Actions from initial state: {len(actions)}")

    action = actions[0]
    next_dist = mdp.step(initial_state, action)
    next_state, reward = next_dist.sample()

    print(f"Initial state : {initial_state.state}")
    print(f"Action        : {action}")
    print(f"Next state    : {next_state.state if isinstance(next_state, NonTerminal) else 'Terminal'}")
    print(f"Reward        : {reward:.2f}")

    NW = mdp.n_wealth_bins
    print(f"\nState space   : {NW*3*3*3*3*36:,} states")
    print(f"Action space  : 81 actions per state")
    print(f"Q-values      : {NW*3*3*3*3*36*81:,}")

    # --- Fast solver ---
    print("\nSolving MDP (fast vectorised solver)...")
    t0 = time.time()
    V, PI = solve_mdp_fast(mdp)
    elapsed = time.time() - t0
    print(f"Solved in {elapsed:.2f}s")

    print(f"\nOptimal value  at s0 : {value_at(V, s0):.2f}")
    print(f"Optimal action at s0 : {policy_at(PI, s0)}")

    print(f"\nOptimal first-month policy by starting wealth (loc=1, wi=1, en=2, perf=1):")
    print(f"  {'Wealth':>10}  {'Housing':>7}  {'Invest':>6}  {'Train':>5}  {'Spend':>5}  {'V':>8}")
    for w in range(NW):
        s = StateV3(wealth=w, location=1, work_intensity=1, energy=2, performance=1, time=0)
        a = policy_at(PI, s)
        v = value_at(V, s)
        h_str = {-1: 'down', 0: 'stay', 1: ' up '}[a.housing]
        print(f"  bin {w:>2} (${w*wu_k:>4}k)  {h_str}     {a.invest}       {a.training}      {a.consumption}   {v:>8.2f}")


if __name__ == '__main__':
    example_usage()
