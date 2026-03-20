from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable, Generic, TypeVar, Callable
from abc import ABC, abstractmethod
import numpy as np
from tqdm.auto import tqdm


# ============================================================================
# Minimal MDP Framework (unchanged from v4)
# ============================================================================

S = TypeVar("S")
A = TypeVar("A")


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
        action: A,
    ) -> Distribution[Tuple[State[S], float]]:
        pass


# ============================================================================
# State / Action v5
# ============================================================================

@dataclass(frozen=True)
class StateV5:
    """
    10D state (v4 + injury):
      cash           {0,...,20}  : $2.5k bins, clipped at $50k
      assets         {0,...,30}  : $5k bins, clipped at $150k
      work_intensity {0,1,2}     : exogenous work demands
      energy         {0,1,2}     : physical energy
      strength       {0,1,2,3}   : strength fitness
      work_cap       {0,1,2,3}   : metabolic/mixed-modal capacity
      endurance      {0,1,2,3}   : aerobic capacity
      load           {0,1,2,3,4} : accumulated training fatigue
      injury         {0,1,2}     : 0=healthy, 1=minor injury, 2=major injury (NEW)
      time           {0,...,35}  : month in 3-year horizon
    """
    cash: int
    assets: int
    work_intensity: int
    energy: int
    strength: int
    work_cap: int
    endurance: int
    load: int
    injury: int
    time: int

    def __lt__(self, other: "StateV5") -> bool:
        return (
            self.cash, self.assets, self.work_intensity, self.energy,
            self.strength, self.work_cap, self.endurance, self.load,
            self.injury, self.time
        ) < (
            other.cash, other.assets, other.work_intensity, other.energy,
            other.strength, other.work_cap, other.endurance, other.load,
            other.injury, other.time
        )


@dataclass(frozen=True)
class ActionV5:
    invest: int       # {0,1,2} : 0%, 10%, 20%
    mode: int         # {0,1,2,3}: recovery, strength, mixed/WC, endurance
    volume: int       # {0,1,2}
    consumption: int  # {0,1,2}


# ============================================================================
# MDP
# ============================================================================

class MultiCapitalMDPv5(MarkovDecisionProcess[StateV5, ActionV5]):
    """
    MDP for multi-capital life optimization (v6-style fitness dynamics on top of v5).

    Changes from v4:
    - New injury state J ∈ {0=healthy, 1=minor, 2=major}
    - Energy and load transitions now conditioned on injury
    - Fitness improvement probability penalised by injury severity
    - New injury transition dynamics driven by injury risk score ρ
    - Reward gains injury penalty and training-while-injured penalty
    - Terminal reward gains injury penalty term

    Neighborhood is fixed at construction time:
      0 = budget, 1 = mid-tier, 2 = premium
    """

    WORK_INTENSITY_MATRIX = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6],
    ], dtype=np.float64)

    def __init__(
        self,
        gamma: float = 0.99,
        time_horizon: int = 36,
        random_seed: int = 42,
        neighborhood: int = 1,
        # monthly income
        y0: float = 11_667.0,
        g: float = 500.0,
        # neighborhood-specific housing costs
        housing_costs: Tuple[float, float, float] = (1_500.0, 2_200.0, 3_000.0),
        base_expense: float = 1_800.0,
        spend_costs: Tuple[float, float, float] = (1_000.0, 2_000.0, 3_500.0),
        # investment
        invest_fracs: Tuple[float, float, float] = (0.0, 0.10, 0.20),
        asset_return: float = 0.005,
        # discretization
        cash_unit: float = 2_500.0,
        asset_unit: float = 5_000.0,
        n_cash_bins: int = 21,
        n_asset_bins: int = 31,
        # utility of discretionary consumption
        consume_utils: Tuple[float, float, float] = (0.0, 1.0, 2.0),
        # reward weights (unchanged from v4)
        alpha_cash: float = 2.5,
        alpha_asset: float = 1.5,
        alpha_str: float = 2.0,
        alpha_wc: float = 2.0,
        alpha_end: float = 2.0,
        alpha_energy: float = 2.0,
        alpha_consumption: float = 1.0,
        alpha_fatigue_energy: float = 1.5,
        alpha_fatigue_work: float = 1.0,
        alpha_fatigue_load: float = 1.0,
        alpha_liquidation: float = 2.0,
        # injury reward weights 
        alpha_inj1: float = 2.0,       # penalty for minor injury
        alpha_inj2: float = 5.0,       # penalty for major injury
        alpha_traininj: float = 2.0,   # penalty for training while injured
        # terminal weights 
        terminal_cash: float = 8.0,
        terminal_asset: float = 15.0,
        terminal_str: float = 10.0,
        terminal_wc: float = 10.0,
        terminal_end: float = 10.0,
        # terminal injury penalty 
        terminal_injury: float = 5.0,
        # v6 fitness dynamics tuning (new v6)
        decay_strength: float = 0.015,
        decay_work_cap: float = 0.05,
        decay_endurance: float = 0.03,
        maintenance_vol_strength: int = 0,
        maintenance_vol_work_cap: int = 1,
        maintenance_vol_endurance: int = 1,
        maintenance_stim_strength: float = 1.4,
        maintenance_stim_work_cap: float = 1.5,
        maintenance_stim_endurance: float = 1.4,
        interference_strength_from_end: float = 0.05,
        interference_endurance_from_strength: float = 0.06,
        interference_workcap_from_specialization: float = 0.06,
    ):
        self.gamma = gamma
        self.time_horizon = time_horizon
        self.rng = np.random.RandomState(random_seed)

        if neighborhood not in (0, 1, 2):
            raise ValueError("neighborhood must be one of {0, 1, 2}")
        self.neighborhood = neighborhood

        self.y0 = y0
        self.g = g
        self.housing_costs = tuple(housing_costs)
        self.base_expense = base_expense
        self.spend_costs = tuple(spend_costs)

        self.invest_fracs = tuple(invest_fracs)
        self.asset_return = asset_return

        self.cash_unit = cash_unit
        self.asset_unit = asset_unit
        self.n_cash_bins = n_cash_bins
        self.n_asset_bins = n_asset_bins

        self.consume_utils = tuple(consume_utils)

        self.alpha = {
            "cash": alpha_cash,
            "asset": alpha_asset,
            "str": alpha_str,
            "wc": alpha_wc,
            "end": alpha_end,
            "energy": alpha_energy,
            "consumption": alpha_consumption,
            "fatigue_energy": alpha_fatigue_energy,
            "fatigue_work": alpha_fatigue_work,
            "fatigue_load": alpha_fatigue_load,
            "liquidation": alpha_liquidation,
            "inj1": alpha_inj1,
            "inj2": alpha_inj2,
            "traininj": alpha_traininj,
        }

        self.terminal_cash = terminal_cash
        self.terminal_asset = terminal_asset
        self.terminal_str = terminal_str
        self.terminal_wc = terminal_wc
        self.terminal_end = terminal_end
        self.terminal_injury = terminal_injury

        self.decay_base = {
            "str": float(decay_strength),
            "wc": float(decay_work_cap),
            "end": float(decay_endurance),
        }
        self.maintenance_min_vol = {
            "str": int(maintenance_vol_strength),
            "wc": int(maintenance_vol_work_cap),
            "end": int(maintenance_vol_endurance),
        }
        self.maintenance_min_stim = {
            "str": float(maintenance_stim_strength),
            "wc": float(maintenance_stim_work_cap),
            "end": float(maintenance_stim_endurance),
        }
        self.interference = {
            "str_from_end": float(interference_strength_from_end),
            "end_from_str": float(interference_endurance_from_strength),
            "wc_from_spec": float(interference_workcap_from_specialization),
        }

    # ------------------------------------------------------------------
    # MDP interface
    # ------------------------------------------------------------------

    def actions(self, state: NonTerminal[StateV5]) -> Iterable[ActionV5]:
        return [
            ActionV5(i, m, v, c)
            for i in range(3)
            for m in range(4)
            for v in range(3)
            for c in range(3)
        ]

    def enumerate_transitions(
        self,
        state: NonTerminal[StateV5],
        action: ActionV5,
    ) -> list[tuple[State[StateV5], float, float]]:
        s = state.state

        next_cash, next_assets, liquidated_amt = self._next_financial_state(
            s.cash, s.assets, action.invest, action.consumption, s.time
        )
        reward = self._compute_reward(s, action, liquidated_amt)

        if s.time >= self.time_horizon - 1:
            return [(Terminal(s), 1.0, reward + self._terminal_reward(s))]

        new_time = s.time + 1
        wi_probs  = self.WORK_INTENSITY_MATRIX[s.work_intensity]
        en_probs  = self._energy_probs_formula(s.energy, s.load, s.work_intensity, action.volume, s.injury)
        ld_probs  = self._load_probs_formula(s.load, s.energy, s.work_intensity, action.mode, action.volume, s.injury)
        str_probs = self._fitness_probs_formula(s.strength, "str", s.energy, s.load, action.mode, action.volume, s.injury)
        wc_probs  = self._fitness_probs_formula(s.work_cap,  "wc",  s.energy, s.load, action.mode, action.volume, s.injury)
        end_probs = self._fitness_probs_formula(s.endurance, "end", s.energy, s.load, action.mode, action.volume, s.injury)
        inj_probs = self._injury_probs_formula(s.injury, s.energy, s.load, s.work_intensity, action.volume)

        transitions = []
        for wi in range(3):
            for en in range(3):
                for ld in range(5):
                    for st in range(4):
                        for wc in range(4):
                            for ed in range(4):
                                for inj in range(3):
                                    prob = (
                                        wi_probs[wi]
                                        * en_probs[en]
                                        * ld_probs[ld]
                                        * str_probs[st]
                                        * wc_probs[wc]
                                        * end_probs[ed]
                                        * inj_probs[inj]
                                    )
                                    if prob < 1e-12:
                                        continue
                                    ns = StateV5(
                                        cash=next_cash,
                                        assets=next_assets,
                                        work_intensity=wi,
                                        energy=en,
                                        strength=st,
                                        work_cap=wc,
                                        endurance=ed,
                                        load=ld,
                                        injury=inj,
                                        time=new_time,
                                    )
                                    transitions.append((NonTerminal(ns), float(prob), reward))
        return transitions

    def step(
        self,
        state: NonTerminal[StateV5],
        action: ActionV5,
    ) -> Distribution[Tuple[State[StateV5], float]]:
        s = state.state

        def sample_transition() -> Tuple[State[StateV5], float]:
            next_cash, next_assets, liquidated_amt = self._next_financial_state(
                s.cash, s.assets, action.invest, action.consumption, s.time
            )
            reward = self._compute_reward(s, action, liquidated_amt)

            if s.time >= self.time_horizon - 1:
                return Terminal(s), reward + self._terminal_reward(s)

            next_s = StateV5(
                cash=next_cash,
                assets=next_assets,
                work_intensity=int(
                    self.rng.choice([0, 1, 2], p=self.WORK_INTENSITY_MATRIX[s.work_intensity])
                ),
                energy=int(
                    self.rng.choice(
                        [0, 1, 2],
                        p=self._energy_probs_formula(s.energy, s.load, s.work_intensity, action.volume, s.injury)
                    )
                ),
                strength=int(
                    self.rng.choice(
                        [0, 1, 2, 3],
                        p=self._fitness_probs_formula(s.strength, "str", s.energy, s.load, action.mode, action.volume, s.injury)
                    )
                ),
                work_cap=int(
                    self.rng.choice(
                        [0, 1, 2, 3],
                        p=self._fitness_probs_formula(s.work_cap, "wc", s.energy, s.load, action.mode, action.volume, s.injury)
                    )
                ),
                endurance=int(
                    self.rng.choice(
                        [0, 1, 2, 3],
                        p=self._fitness_probs_formula(s.endurance, "end", s.energy, s.load, action.mode, action.volume, s.injury)
                    )
                ),
                load=int(
                    self.rng.choice(
                        [0, 1, 2, 3, 4],
                        p=self._load_probs_formula(s.load, s.energy, s.work_intensity, action.mode, action.volume, s.injury)
                    )
                ),
                injury=int(
                    self.rng.choice(
                        [0, 1, 2],
                        p=self._injury_probs_formula(s.injury, s.energy, s.load, s.work_intensity, action.volume)
                    )
                ),
                time=s.time + 1,
            )
            return NonTerminal(next_s), reward

        return SampledDistribution(sample_transition)

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    def _energy_probs_formula(self, e: int, l: int, wi: int, vol: int, j: int) -> np.ndarray:
        """
        v5 change: injury adds to fatigue pressure and blocks recovery.
        D = vol + 1{wi=2} + 1{l>=3} + 1{j>=1}
        R = 1{vol=0 AND wi!=2 AND j=0}
        """
        D = vol + int(wi == 2) + int(l >= 3) + int(j >= 1)
        R = int(vol == 0 and wi != 2 and j == 0)
        E_tilde = int(np.clip(e - int(D >= 2) + R, 0, 2))
        probs = np.zeros(3)
        probs[E_tilde] += 0.70
        probs[min(E_tilde + 1, 2)] += 0.15
        probs[max(E_tilde - 1, 0)] += 0.15
        return probs

    def _load_probs_formula(self, l: int, e: int, wi: int, mode: int, vol: int, j: int) -> np.ndarray:
        """
        v5 change: injury adds +1 to effective fatigue accumulation.
        L_tilde = clip(l + S + 1{j>=1} - R_rec, 0, 4)
        """
        S = 0 if mode == 0 else (1 + vol)
        R_rec = int(e == 2 and wi != 2)
        L_tilde = int(np.clip(l + S + int(j >= 1) - R_rec, 0, 4))
        probs = np.zeros(5)
        probs[L_tilde] += 0.70
        probs[min(L_tilde + 1, 4)] += 0.15
        probs[max(L_tilde - 1, 0)] += 0.15
        return probs

    def _fitness_probs_formula(
        self,
        f: int,
        k: str,
        e: int,
        l: int,
        mode: int,
        vol: int,
        j: int
    ) -> np.ndarray:
        """
        Revised fitness dynamics with:
        1) easy early gains ("novice gains")
        2) maintenance requirements that matter mainly at higher fitness
        3) diminishing returns near max fitness
        4) mild cross-interference
        5) domain-specific decay (wc fastest, end moderate, str slowest)

        Modes:
        0 = recovery
        1 = strength
        2 = mixed / work-capacity
        3 = endurance
        """

        # --------------------------------------------------------------
        # 1. Domain-specific stimulus mapping
        # --------------------------------------------------------------
        if k == "str":
            strong_target = int(mode == 1)
            partial_target = int(mode == 2)
            T_k = (
                2.4 + 0.9 * vol if strong_target else
                1.2 + 0.5 * vol if partial_target else
                0.0
            )
            interference = (self.interference["str_from_end"] + 0.01 * vol) if mode == 3 else 0.0

        elif k == "wc":
            strong_target = int(mode == 2)
            partial_target = int(mode in (1, 3))
            T_k = (
                2.5 + 0.9 * vol if strong_target else
                1.0 + 0.4 * vol if partial_target else
                0.0
            )
            interference = (self.interference["wc_from_spec"] + 0.01 * vol) if mode in (1, 3) else 0.0

        else:  # k == "end"
            strong_target = int(mode == 3)
            partial_target = int(mode == 2)
            T_k = (
                2.4 + 0.9 * vol if strong_target else
                1.2 + 0.5 * vol if partial_target else
                0.0
            )
            interference = (self.interference["end_from_str"] + 0.01 * vol) if mode == 1 else 0.0

        # --------------------------------------------------------------
        # 2. Recovery / readiness
        # --------------------------------------------------------------
        good_recovery = int(e >= 1 and l <= 2 and j == 0)
        okay_recovery = int(e >= 1 and l <= 3 and j <= 1)
        bad_recovery = int(e == 0 or l >= 4 or j >= 1)

        # --------------------------------------------------------------
        # 3. Diminishing returns
        #    Make early gains easier, 2->3 noticeably harder.
        # --------------------------------------------------------------
        gain_multiplier = [1.00, 0.90, 0.55, 0.00][f]

        # --------------------------------------------------------------
        # 4. Improvement probability
        #    Crucial change: strong novice gains at low fitness.
        # --------------------------------------------------------------
        novice_bonus = [0.18, 0.08, 0.00, 0.00][f]

        p_gain_raw = (
            novice_bonus
            + 0.08
            + 0.10 * T_k
            + 0.05 * good_recovery
            + 0.02 * okay_recovery
            - 0.04 * int(e == 0)
            - 0.03 * int(l >= 4)
            - 0.06 * int(j == 1)
            - 0.12 * int(j == 2)
            - 0.03 * interference
        )

        p_gain = float(np.clip(gain_multiplier * p_gain_raw, 0.0, 0.75))

        # --------------------------------------------------------------
        # 5. Maintenance requirement
        #    Important change: maintenance only really matters once fitness
        #    is already somewhat developed.
        # --------------------------------------------------------------
        if f <= 1:
            maint_ok = True
        else:
            maint_ok = (
                vol >= self.maintenance_min_vol[k]
                and T_k >= self.maintenance_min_stim[k]
                and e >= 1
                and l <= 3
                and j <= 1
            )

        # --------------------------------------------------------------
        # 6. Domain-specific decay
        #    Important change: low-fitness states decay very little.
        # --------------------------------------------------------------
        if f == 0:
            d_k = 0.0
        elif f == 1:
            d_k = 0.25 * self.decay_base[k]
        else:
            d_k = self.decay_base[k]

        # Recovery mode causes some decay, but not catastrophic decay
        if mode == 0:
            d_k += 0.03

        # Not enough stimulus mainly matters at higher fitness
        if f >= 2:
            if not maint_ok:
                d_k += 0.04
            if T_k == 0:
                d_k += 0.04
            elif T_k < self.maintenance_min_stim[k]:
                d_k += 0.02
        else:
            # At low fitness, lack of stimulus should not cause collapse
            if T_k == 0 and mode == 0:
                d_k += 0.01

        # Fatigue / injury raise decay some
        if e == 0:
            d_k += 0.02
        if l >= 4:
            d_k += 0.02
        if j == 1:
            d_k += 0.03
        elif j == 2:
            d_k += 0.06

        # Mild interference
        d_k += interference

        # Adequate targeted training protects fitness
        if maint_ok:
            if strong_target:
                d_k -= 0.03
            elif partial_target:
                d_k -= 0.01

        # Top-end fitness is harder to hold
        if f == 3:
            d_k += 0.03

        d_k = float(np.clip(d_k, 0.0, 0.35))

        # --------------------------------------------------------------
        # 7. Normalize
        # --------------------------------------------------------------
        if p_gain + d_k > 1.0:
            total = p_gain + d_k
            p_gain /= total
            d_k /= total

        probs = np.zeros(4)
        probs[min(f + 1, 3)] += p_gain
        probs[max(f - 1, 0)] += d_k
        probs[f] += max(0.0, 1.0 - p_gain - d_k)
        return probs

    def _injury_risk_score(self, e: int, l: int, wi: int, vol: int) -> float:
        """
        ρ = min(1, 0.05 + 0.15*1{l>=3} + 0.10*1{e=0} + 0.10*1{wi=2} + 0.10*1{vol=2})
        Note: does not depend on mode.
        """
        return min(1.0,
                   0.05
                   + 0.15 * int(l >= 3)
                   + 0.10 * int(e == 0)
                   + 0.10 * int(wi == 2)
                   + 0.10 * int(vol == 2))

    def _injury_probs_formula(self, j: int, e: int, l: int, wi: int, vol: int) -> np.ndarray:
        """
        Injury transition P(J' | J=j, ρ).
          J=0: P(0)=1-ρ, P(1)=ρ, P(2)=0
          J=1: P(0)=0.40, P(1)=0.50, P(2)=0.10
          J=2: P(0)=0.10, P(1)=0.30, P(2)=0.60
        """
        if j == 0:
            rho = self._injury_risk_score(e, l, wi, vol)
            return np.array([1.0 - rho, rho, 0.0])
        elif j == 1:
            return np.array([0.40, 0.50, 0.10])
        else:  # j == 2
            return np.array([0.10, 0.30, 0.60])

    # ------------------------------------------------------------------
    # Financial helpers (unchanged from v4)
    # ------------------------------------------------------------------

    def _income(self, t: int) -> float:
        return self.y0 + self.g * (t // 12)

    def _housing_cost(self) -> float:
        return self.housing_costs[self.neighborhood]

    def _cash_bin_to_dollars(self, c: int) -> float:
        return c * self.cash_unit

    def _asset_bin_to_dollars(self, a: int) -> float:
        return a * self.asset_unit

    def _dollars_to_cash_bin(self, dollars: float) -> int:
        return int(np.clip(np.round(dollars / self.cash_unit), 0, self.n_cash_bins - 1))

    def _dollars_to_asset_bin(self, dollars: float) -> int:
        return int(np.clip(np.round(dollars / self.asset_unit), 0, self.n_asset_bins - 1))

    def _next_financial_state(
        self,
        cash_bin: int,
        asset_bin: int,
        invest: int,
        consumption: int,
        t: int,
    ) -> tuple[int, int, float]:
        """Returns: next_cash_bin, next_asset_bin, liquidated_amount_dollars"""
        cash = self._cash_bin_to_dollars(cash_bin)
        assets = self._asset_bin_to_dollars(asset_bin)

        income = self._income(t)
        required = self._housing_cost() + self.base_expense
        discretionary = self.spend_costs[consumption]

        liquid_funds = cash + income - required - discretionary
        invest_amt = self.invest_fracs[invest] * max(liquid_funds, 0.0)
        cash_hat = liquid_funds - invest_amt
        assets_hat = assets + invest_amt
        assets_after_return = assets_hat * (1.0 + self.asset_return)
        liquidation = min(assets_after_return, max(-cash_hat, 0.0))
        cash_next = cash_hat + liquidation
        assets_next = assets_after_return - liquidation

        return (
            self._dollars_to_cash_bin(cash_next),
            self._dollars_to_asset_bin(assets_next),
            float(liquidation),
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        state: StateV5,
        action: ActionV5,
        liquidated_amt: float,
    ) -> float:
        a = self.alpha
        fatigue_e  = int(state.energy == 0 and action.volume == 2)
        fatigue_w  = int(state.work_intensity == 2 and action.volume == 2)
        fatigue_l  = int(state.load >= 3 and action.mode != 0 and action.volume >= 1)
        liq_pen    = float(liquidated_amt > 0.0)
        inj_pen    = (a["inj1"] * int(state.injury == 1)
                      + a["inj2"] * int(state.injury == 2))
        traininj   = float(state.injury >= 1 and action.mode != 0 and action.volume >= 1)

        return float(
            a["cash"]  * np.sqrt(state.cash + 1.0)
            + a["asset"] * np.sqrt(state.assets + 1.0)
            + a["str"]   * state.strength
            + a["wc"]    * state.work_cap
            + a["end"]   * state.endurance
            + a["energy"] * state.energy
            + a["consumption"] * self.consume_utils[action.consumption]
            - a["fatigue_energy"] * fatigue_e
            - a["fatigue_work"]   * fatigue_w
            - a["fatigue_load"]   * fatigue_l
            - a["liquidation"]    * liq_pen
            - inj_pen
            - a["traininj"] * traininj
        )

    def _terminal_reward(self, state: StateV5) -> float:
        return float(
            self.terminal_cash   * state.cash
            + self.terminal_asset  * state.assets
            + self.terminal_str    * state.strength
            + self.terminal_wc     * state.work_cap
            + self.terminal_end    * state.endurance
            - self.terminal_injury * state.injury
        )


# ============================================================================
# Solver
# ============================================================================

def _compute_K_kernel_v5(
    V_next: np.ndarray,
    wi_mat: np.ndarray,
    en_table: np.ndarray,
    load_table: np.ndarray,
    str_table: np.ndarray,
    wc_table: np.ndarray,
    end_table: np.ndarray,
    inj_table: np.ndarray,
    mode: int,
    vol: int,
) -> np.ndarray:
    """
    Compute the expected-value kernel K for a fixed (mode, vol) pair.

    V_next axes: (NC, NA, NWI', NE', NL', NS', NWC', NEND', NJ')
    Output shape: same as V_next = (NC, NA, NWI, NE, NL, NS, NWC, NEND, NJ)

    Outer loop: (wi_c, e_c, l_c, j_c) — 135 iterations.
    For each iteration, 7 sequential einsums marginalise all stochastic next-state dims.
    """
    NC, NA, NWI, NE, NL, NS, NWC, NEND, NJ = V_next.shape
    K = np.zeros_like(V_next)

    for wi_c in range(NWI):
        p_wi = wi_mat[wi_c].astype(np.float32)                       # (NWI,)
        for e_c in range(NE):
            for l_c in range(NL):
                for j_c in range(NJ):
                    p_load    = load_table[wi_c, e_c, l_c, j_c, mode, vol, :].astype(np.float32)  # (NL,)
                    p_en_vec  = en_table[wi_c, e_c, l_c, j_c, vol, :].astype(np.float32)          # (NE,)
                    p_inj_vec = inj_table[j_c, e_c, l_c, wi_c, vol, :].astype(np.float32)         # (NJ,)
                    p_str = str_table[:, e_c, l_c, j_c, mode, vol, :].astype(np.float32)  # (NS, NS)
                    p_wc  = wc_table[:, e_c, l_c, j_c, mode, vol, :].astype(np.float32)   # (NWC, NWC)
                    p_end = end_table[:, e_c, l_c, j_c, mode, vol, :].astype(np.float32)  # (NEND, NEND)

                    # V_next axes: (c, a, W, E, L, S, X, Q, J)
                    #   W=NWI', E=NE', L=NL', S=NS', X=NWC', Q=NEND', J=NJ'

                    # 1. Marginalise NL' with p_load
                    V1 = np.einsum('caWELSXQJ,L->caWESXQJ', V_next, p_load, optimize=True)

                    # 2. Marginalise NJ' with p_inj_vec (independent of fitness dims)
                    V2 = np.einsum('caWESXQJ,J->caWESXQ', V1, p_inj_vec, optimize=True)

                    # 3. Marginalise NEND' (Q) with p_end (r=NEND_c, Q=NEND')
                    V3 = np.einsum('caWESXQ,rQ->caWESXr', V2, p_end, optimize=True)

                    # 4. Marginalise NWC' (X) with p_wc (p=NWC_c, X=NWC')
                    V4 = np.einsum('caWESXr,pX->caWESpr', V3, p_wc, optimize=True)

                    # 5. Marginalise NS' (S) with p_str (s=NS_c, S=NS')
                    V5 = np.einsum('caWESpr,sS->caWEspr', V4, p_str, optimize=True)

                    # 6. Marginalise NE' (E) with p_en_vec
                    V6 = np.einsum('caWEspr,E->caWspr', V5, p_en_vec, optimize=True)

                    # 7. Marginalise NWI' (W) with p_wi
                    V7 = np.einsum('caWspr,W->caspr', V6, p_wi, optimize=True)

                    # V7 shape: (NC, NA, NS_c, NWC_c, NEND_c)
                    K[:, :, wi_c, e_c, l_c, :, :, :, j_c] = V7

    return K


def solve_mdp_fast_v5(mdp: MultiCapitalMDPv5) -> tuple[np.ndarray, np.ndarray]:
    """
    Backward induction for v5 MDP.

    Returns:
        V_all  : shape (NC, NA, NWI, NE, NL, NS, NWC, NEND, NJ, T)  float32
        PI_all : shape (NC, NA, NWI, NE, NL, NS, NWC, NEND, NJ, T, 4)  int8
                 last axis = [invest, mode, volume, consumption]
    """
    T    = mdp.time_horizon
    NC   = mdp.n_cash_bins   # 21
    NA   = mdp.n_asset_bins  # 31
    NWI  = 3
    NE   = 3
    NL   = 5
    NS   = 4
    NWC  = 4
    NEND = 4
    NJ   = 3

    wi_mat = mdp.WORK_INTENSITY_MATRIX

    # ---- Precompute transition tables ----

    # Energy: (NWI, NE, NL, NJ, vol=3, NE')
    en_table = np.zeros((NWI, NE, NL, NJ, 3, NE), dtype=np.float64)
    for wi in range(NWI):
        for e in range(NE):
            for l in range(NL):
                for j in range(NJ):
                    for v in range(3):
                        en_table[wi, e, l, j, v] = mdp._energy_probs_formula(e, l, wi, v, j)

    # Load: (NWI, NE, NL, NJ, mode=4, vol=3, NL')
    load_table = np.zeros((NWI, NE, NL, NJ, 4, 3, NL), dtype=np.float64)
    for wi in range(NWI):
        for e in range(NE):
            for l in range(NL):
                for j in range(NJ):
                    for m in range(4):
                        for v in range(3):
                            load_table[wi, e, l, j, m, v] = mdp._load_probs_formula(l, e, wi, m, v, j)

    # Fitness: (f=4, NE, NL, NJ, mode=4, vol=3, f'=4)
    str_table  = np.zeros((NS,   NE, NL, NJ, 4, 3, NS),   dtype=np.float64)
    wc_table   = np.zeros((NWC,  NE, NL, NJ, 4, 3, NWC),  dtype=np.float64)
    end_table  = np.zeros((NEND, NE, NL, NJ, 4, 3, NEND), dtype=np.float64)
    for f in range(4):
        for e in range(NE):
            for l in range(NL):
                for j in range(NJ):
                    for m in range(4):
                        for v in range(3):
                            str_table[f, e, l, j, m, v]  = mdp._fitness_probs_formula(f, "str", e, l, m, v, j)
                            wc_table[f, e, l, j, m, v]   = mdp._fitness_probs_formula(f, "wc",  e, l, m, v, j)
                            end_table[f, e, l, j, m, v]  = mdp._fitness_probs_formula(f, "end", e, l, m, v, j)

    # Injury: (NJ, NE, NL, NWI, vol=3, NJ')  — no mode dependence
    inj_table = np.zeros((NJ, NE, NL, NWI, 3, NJ), dtype=np.float64)
    for j in range(NJ):
        for e in range(NE):
            for l in range(NL):
                for wi in range(NWI):
                    for v in range(3):
                        inj_table[j, e, l, wi, v] = mdp._injury_probs_formula(j, e, l, wi, v)

    # ---- Precompute financial tables (unchanged from v4) ----
    fin_next_c = np.zeros((NC, NA, 3, 3, T), dtype=np.int16)
    fin_next_a = np.zeros((NC, NA, 3, 3, T), dtype=np.int16)
    fin_liq    = np.zeros((NC, NA, 3, 3, T), dtype=np.float32)

    for t in range(T):
        for c in range(NC):
            for a_bin in range(NA):
                for inv in range(3):
                    for cons in range(3):
                        nc, na, liq = mdp._next_financial_state(c, a_bin, inv, cons, t)
                        fin_next_c[c, a_bin, inv, cons, t] = nc
                        fin_next_a[c, a_bin, inv, cons, t] = na
                        fin_liq[c, a_bin, inv, cons, t]    = liq

    # ---- Precompute state reward — 9D broadcast ----
    # State-dependent components that don't vary with actions.
    # Shape: (NC, NA, 1, NE, 1, NS, NWC, NEND, NJ) broadcasts to STATE_SHAPE.
    a = mdp.alpha
    C9   = np.arange(NC,   dtype=np.float32)[:, None, None, None, None, None, None, None, None]
    A9   = np.arange(NA,   dtype=np.float32)[None, :, None, None, None, None, None, None, None]
    E9   = np.arange(NE,   dtype=np.float32)[None, None, None, :, None, None, None, None, None]
    S9   = np.arange(NS,   dtype=np.float32)[None, None, None, None, None, :, None, None, None]
    WC9  = np.arange(NWC,  dtype=np.float32)[None, None, None, None, None, None, :, None, None]
    END9 = np.arange(NEND, dtype=np.float32)[None, None, None, None, None, None, None, :, None]
    J9   = np.arange(NJ,   dtype=np.float32)[None, None, None, None, None, None, None, None, :]
    WI9  = np.arange(NWI,  dtype=np.float32)[None, None, :, None, None, None, None, None, None]
    L9   = np.arange(NL,   dtype=np.float32)[None, None, None, None, :, None, None, None, None]

    # Injury penalty (state-dependent, action-independent)
    inj_pen_state = (a["inj1"] * (J9 == 1).astype(np.float32)
                     + a["inj2"] * (J9 == 2).astype(np.float32))

    R_state = (
        a["cash"]  * np.sqrt(C9 + 1.0)
        + a["asset"] * np.sqrt(A9 + 1.0)
        + a["energy"] * E9
        + a["str"]    * S9
        + a["wc"]     * WC9
        + a["end"]    * END9
        - inj_pen_state
    ).astype(np.float32)
    # Shape broadcasts to (NC, NA, 1, NE, 1, NS, NWC, NEND, NJ)

    # ---- Allocate output arrays ----
    STATE_SHAPE = (NC, NA, NWI, NE, NL, NS, NWC, NEND, NJ)
    V_all  = np.zeros(STATE_SHAPE + (T,),    dtype=np.float32)
    PI_all = np.zeros(STATE_SHAPE + (T, 4),  dtype=np.int8)
    V_next = np.zeros(STATE_SHAPE,            dtype=np.float32)

    # ---- Backward induction ----
    for t in tqdm(reversed(range(T)), total=T, desc="Backward induction"):
        Q_best = np.full(STATE_SHAPE, -np.inf, dtype=np.float32)
        PI_t   = np.zeros(STATE_SHAPE + (4,), dtype=np.int8)

        # Pre-compute K kernels for each (mode, vol) pair — skip at terminal step
        K_kernels = {}
        if t < T - 1:
            for mode in range(4):
                for vol in range(3):
                    K_kernels[(mode, vol)] = _compute_K_kernel_v5(
                        V_next, wi_mat,
                        en_table, load_table,
                        str_table, wc_table, end_table,
                        inj_table,
                        mode, vol,
                    )

        for inv in range(3):
            for mode in range(4):
                for vol in range(3):
                    # Fatigue penalties (state-dependent, action-fixed for this (mode,vol))
                    fat_e = (a["fatigue_energy"] * float(vol == 2)
                             * (E9 == 0).astype(np.float32))
                    fat_w = (a["fatigue_work"] * float(vol == 2)
                             * (WI9 == 2).astype(np.float32))
                    fat_l = (a["fatigue_load"]
                             * float(mode != 0 and vol >= 1)
                             * (L9 >= 3).astype(np.float32))
                    # Training-while-injured penalty
                    train_inj = (a["traininj"]
                                 * float(mode != 0 and vol >= 1)
                                 * (J9 >= 1).astype(np.float32))

                    K = K_kernels.get((mode, vol))  # None at terminal step

                    for cons in range(3):
                        next_c = fin_next_c[:, :, inv, cons, t]   # (NC, NA)
                        next_a = fin_next_a[:, :, inv, cons, t]   # (NC, NA)
                        liq_pen = (
                            a["liquidation"]
                            * (fin_liq[:, :, inv, cons, t] > 0.0).astype(np.float32)
                        )  # (NC, NA)

                        # Broadcast liq_pen to 9D
                        liq_pen_9d = liq_pen[:, :, None, None, None, None, None, None, None]

                        R_act = (
                            R_state
                            + a["consumption"] * mdp.consume_utils[cons]
                            - fat_e
                            - fat_w
                            - fat_l
                            - train_inj
                            - liq_pen_9d
                        ).astype(np.float32)
                        # R_act shape: (NC, NA, NWI, NE, NL, NS, NWC, NEND, NJ)

                        if t == T - 1:
                            Q = R_act + (
                                mdp.terminal_cash    * next_c[:, :, None, None, None, None, None, None, None].astype(np.float32)
                                + mdp.terminal_asset * next_a[:, :, None, None, None, None, None, None, None].astype(np.float32)
                                + mdp.terminal_str   * S9
                                + mdp.terminal_wc    * WC9
                                + mdp.terminal_end   * END9
                                - mdp.terminal_injury * J9
                            )
                        else:
                            future = K[next_c, next_a, :, :, :, :, :, :, :]
                            Q = R_act + mdp.gamma * future

                        improved = Q > Q_best
                        Q_best = np.where(improved, Q, Q_best)
                        PI_t[..., 0] = np.where(improved, inv,  PI_t[..., 0])
                        PI_t[..., 1] = np.where(improved, mode, PI_t[..., 1])
                        PI_t[..., 2] = np.where(improved, vol,  PI_t[..., 2])
                        PI_t[..., 3] = np.where(improved, cons, PI_t[..., 3])

        V_next = Q_best
        V_all[..., t]    = Q_best
        PI_all[..., t, :] = PI_t

    return V_all, PI_all


# ============================================================================
# Helper functions
# ============================================================================

def policy_at_v5(PI: np.ndarray, s: StateV5) -> ActionV5:
    """Extract optimal action from policy array at state s."""
    raw = PI[s.cash, s.assets, s.work_intensity, s.energy,
             s.load, s.strength, s.work_cap, s.endurance, s.injury, s.time]
    return ActionV5(
        invest=int(raw[0]),
        mode=int(raw[1]),
        volume=int(raw[2]),
        consumption=int(raw[3]),
    )


def value_at_v5(V: np.ndarray, s: StateV5) -> float:
    """Extract value estimate from value array at state s."""
    return float(V[s.cash, s.assets, s.work_intensity, s.energy,
                   s.load, s.strength, s.work_cap, s.endurance, s.injury, s.time])

from pathlib import Path
import json
import numpy as np


def save_solver_outputs_v5(
    filepath: str | Path,
    V: np.ndarray,
    PI: np.ndarray,
    mdp: MultiCapitalMDPv5,
) -> None:
    """
    Save exact-solver outputs and basic MDP metadata to a compressed .npz file.
    """
    filepath = Path(filepath)

    metadata = {
        "time_horizon": mdp.time_horizon,
        "gamma": mdp.gamma,
        "neighborhood": mdp.neighborhood,
        "cash_unit": mdp.cash_unit,
        "asset_unit": mdp.asset_unit,
        "n_cash_bins": mdp.n_cash_bins,
        "n_asset_bins": mdp.n_asset_bins,
        "y0": mdp.y0,
        "g": mdp.g,
        "asset_return": mdp.asset_return,
        "alpha": mdp.alpha,
        "terminal_cash": mdp.terminal_cash,
        "terminal_asset": mdp.terminal_asset,
        "terminal_str": mdp.terminal_str,
        "terminal_wc": mdp.terminal_wc,
        "terminal_end": mdp.terminal_end,
        "terminal_injury": mdp.terminal_injury,
    }

    np.savez_compressed(
        filepath,
        V=V,
        PI=PI,
        metadata_json=json.dumps(metadata),
    )


def load_solver_outputs_v5(
    filepath: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load exact-solver outputs saved by save_solver_outputs_v5.
    Returns:
        V, PI, metadata
    """
    filepath = Path(filepath)
    data = np.load(filepath, allow_pickle=False)

    V = data["V"]
    PI = data["PI"]
    metadata = json.loads(str(data["metadata_json"]))

    return V, PI, metadata