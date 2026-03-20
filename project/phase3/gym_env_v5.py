"""
LifeGymEnv5: gymnasium.Env wrapper around MultiCapitalMDPv5 for SB3 compatibility.

Observation: float32 array shape (114,) consisting of concatenated one-hot encodings:
             [cash(21), assets(31), work_intensity(3), energy(3),
              strength(4), work_cap(4), endurance(4), load(5), injury(3), time(36)]
Action:      MultiDiscrete([3, 4, 3, 3]) = [invest, mode, volume, consumption]
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from multi_capital_mdp_v6 import (
    MultiCapitalMDPv5,
    StateV5,
    ActionV5,
    NonTerminal,
    Terminal,
)

# ---------------------------------------------------------------------------
# Default initial-state distribution
# (master's student starting first job, active competitive weightlifter)
# ---------------------------------------------------------------------------
DEFAULT_INIT_WEIGHTS: dict[str, list[float]] = {
    "cash":           [0.15, 0.30, 0.25, 0.15, 0.10, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0],
    "assets":         [0.25, 0.30, 0.20, 0.15, 0.07, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0],
    "work_intensity": [0.40, 0.45, 0.15],
    "energy":         [0.05, 0.20, 0.75],
    "strength":       [0.05, 0.15, 0.40, 0.40],
    "work_cap":       [0.05, 0.20, 0.45, 0.30],
    "endurance":      [0.10, 0.25, 0.40, 0.25],
    "load":           [0.50, 0.30, 0.15, 0.04, 0.01],
    "injury":         [0.90, 0.08, 0.02],  # mostly healthy at start
}


class LifeGymEnv5(gym.Env):
    """
    Stateful gymnasium wrapper around MultiCapitalMDPv5.

    Parameters
    ----------
    neighborhood : int
        Fixed neighborhood tier (0=budget, 1=mid-tier, 2=premium).
    mdp_kwargs : dict | None
        Extra keyword arguments forwarded to MultiCapitalMDPv5.
    """

    metadata = {"render_modes": []}

    _CASH_DIM     = 21
    _ASSET_DIM    = 31
    _WI_DIM       = 3
    _ENERGY_DIM   = 3
    _STR_DIM      = 4
    _WC_DIM       = 4
    _END_DIM      = 4
    _LOAD_DIM     = 5
    _INJ_DIM      = 3
    _TIME_DIM     = 36
    _OBS_DIM = (
        _CASH_DIM + _ASSET_DIM + _WI_DIM + _ENERGY_DIM
        + _STR_DIM + _WC_DIM + _END_DIM + _LOAD_DIM + _INJ_DIM + _TIME_DIM
    )  # 114

    def __init__(
        self,
        neighborhood: int = 1,
        mdp_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        kwargs = mdp_kwargs or {}
        self.mdp = MultiCapitalMDPv5(neighborhood=neighborhood, **kwargs)
        self.neighborhood = neighborhood

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3, 4, 3, 3])

        def _norm(v: list[float]) -> np.ndarray:
            a = np.array(v, dtype=float)
            if a.sum() <= 0:
                raise ValueError("Initial-state weight vector must have positive sum.")
            return a / a.sum()

        raw = DEFAULT_INIT_WEIGHTS
        self._w_cash   = _norm(raw["cash"])
        self._w_assets = _norm(raw["assets"])
        self._w_wi     = _norm(raw["work_intensity"])
        self._w_energy = _norm(raw["energy"])
        self._w_str    = _norm(raw["strength"])
        self._w_wc     = _norm(raw["work_cap"])
        self._w_end    = _norm(raw["endurance"])
        self._w_load   = _norm(raw["load"])
        self._w_inj    = _norm(raw["injury"])

        self._state: StateV5 | None = None
        self._rng = np.random.default_rng()

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.mdp.rng = np.random.RandomState(seed)

        self._state = StateV5(
            cash=int(self._rng.choice(self._CASH_DIM,  p=self._w_cash)),
            assets=int(self._rng.choice(self._ASSET_DIM, p=self._w_assets)),
            work_intensity=int(self._rng.choice(self._WI_DIM,     p=self._w_wi)),
            energy=int(self._rng.choice(self._ENERGY_DIM, p=self._w_energy)),
            strength=int(self._rng.choice(self._STR_DIM,   p=self._w_str)),
            work_cap=int(self._rng.choice(self._WC_DIM,    p=self._w_wc)),
            endurance=int(self._rng.choice(self._END_DIM,   p=self._w_end)),
            load=int(self._rng.choice(self._LOAD_DIM,  p=self._w_load)),
            injury=int(self._rng.choice(self._INJ_DIM,   p=self._w_inj)),
            time=0,
        )
        return self._state_to_obs(self._state), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None, "Call reset() before step()."

        act = ActionV5(
            invest=int(action[0]),
            mode=int(action[1]),
            volume=int(action[2]),
            consumption=int(action[3]),
        )

        dist = self.mdp.step(NonTerminal(self._state), act)
        next_wrapped, reward = dist.sample()

        terminated = isinstance(next_wrapped, Terminal)
        self._state = next_wrapped.state

        return self._state_to_obs(self._state), float(reward), terminated, False, {}

    def render(self) -> None:
        pass

    @staticmethod
    def _one_hot(index: int, size: int) -> np.ndarray:
        v = np.zeros(size, dtype=np.float32)
        v[index] = 1.0
        return v

    def _state_to_obs(self, state: StateV5) -> np.ndarray:
        return np.concatenate([
            self._one_hot(state.cash,           self._CASH_DIM),
            self._one_hot(state.assets,         self._ASSET_DIM),
            self._one_hot(state.work_intensity, self._WI_DIM),
            self._one_hot(state.energy,         self._ENERGY_DIM),
            self._one_hot(state.strength,       self._STR_DIM),
            self._one_hot(state.work_cap,       self._WC_DIM),
            self._one_hot(state.endurance,      self._END_DIM),
            self._one_hot(state.load,           self._LOAD_DIM),
            self._one_hot(state.injury,         self._INJ_DIM),
            self._one_hot(state.time,           self._TIME_DIM),
        ]).astype(np.float32)
