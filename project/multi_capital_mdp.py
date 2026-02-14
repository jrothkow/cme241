"""
Multi-Capital Decision-Making Under Energy Constraints

This module implements a Markov Decision Process for optimizing life decisions
across multiple capital types (financial, human, energy) for an early-career
professional who is also a competitive athlete.

The MDP can be parameterized to represent three versions:
- V3: Simplified discrete version (solvable with exact DP)
- V2: Realistic mixed discrete/continuous version (solvable with deep RL)
- V1: Ideal continuous version (research-level complexity)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable, Generic, TypeVar, Callable
from abc import ABC, abstractmethod
import numpy as np


# ============================================================================
# Minimal MDP Framework (subset of course materials for Phase 1)
# ============================================================================

S = TypeVar('S')
A = TypeVar('A')


class State(ABC, Generic[S]):
    """Abstract base class for states"""
    state: S


@dataclass(frozen=True)
class Terminal(State[S]):
    """Terminal state"""
    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    """Non-terminal state"""
    state: S

    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return self.state < other.state


class Distribution(ABC, Generic[S]):
    """Abstract distribution"""
    @abstractmethod
    def sample(self) -> S:
        pass


class SampledDistribution(Distribution[S]):
    """Distribution defined by a sampling function"""
    def __init__(self, sampler: Callable[[], S]):
        self.sampler = sampler

    def sample(self) -> S:
        return self.sampler()


class MarkovDecisionProcess(ABC, Generic[S, A]):
    """Abstract MDP base class"""
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
# Version 3: Simplified DP Version (Phase 2)
# ============================================================================

@dataclass(frozen=True)
class StateV3:
    """State for Version 3: Fully discrete, 9-dimensional state space

    This version is tractable with exact dynamic programming.
    """
    wealth: int  # {0, 1, ..., 10}: Wealth level in units of $10k
    emergency: int  # {0, 1, 2}: Emergency fund level (none/partial/full)
    location: int  # {0, 1, 2}: Neighborhood tier (cheap-far/medium/expensive-close)
    work_intensity: int  # {0, 1, 2}: Work intensity (low/medium/high, stochastic)
    energy: int  # {0, 1, 2}: Physical energy level (low/medium/high)
    strength: int  # {0, 1, 2}: Strength level (beginner/intermediate/advanced)
    salary: int  # {0, 1, 2}: Salary tier ($60k/$80k/$100k+)
    market: int  # {0, 1}: Market regime (bear/bull)
    time: int  # {0, 1, ..., 35}: Month (3-year horizon)

    def __lt__(self, other: StateV3) -> bool:
        """Enable sorting for state enumeration in DP"""
        return (self.wealth, self.emergency, self.location, self.work_intensity,
                self.energy, self.strength, self.salary, self.market, self.time) < \
               (other.wealth, other.emergency, other.location, other.work_intensity,
                other.energy, other.strength, other.salary, other.market, other.time)


@dataclass(frozen=True)
class ActionV3:
    """Action for Version 3: Fully discrete, 4-dimensional action space"""
    housing: int  # {-1, 0, 1}: Downgrade/stay/upgrade neighborhood
    invest: int  # {0, 1, 2}: Investment level (0%, 10%, 20% of income)
    training: int  # {0, 1, 2}: Training intensity (light/moderate/intense)
    consumption: int  # {0, 1, 2}: Discretionary spending (frugal/moderate/luxury)


class MultiCapitalEnergyMDP(MarkovDecisionProcess[StateV3, ActionV3]):
    """
    MDP for multi-capital optimization under energy constraints.

    This class can be parameterized to represent three versions:
    - version='v3': Simplified discrete (Phase 2, DP-solvable)
    - version='v2': Realistic mixed (Phase 3, RL-solvable)
    - version='v1': Ideal continuous

    For Phase 1, we implement V3. V2 and V1 would extend this structure
    with continuous state spaces and more complex transition dynamics.
    """

    def __init__(
        self,
        version: str = 'v3',
        gamma: float = 0.99,
        time_horizon: int = 36,  # months for V3
        initial_salary_tier: int = 0,  # Start at $60k
        work_location: Tuple[float, float] = (40.04, -75.52),  # Malvern coords
        gym_location: Tuple[float, float] = (40.02, -75.50),  # Example gym
        random_seed: int = 42
    ):
        self.version = version
        self.gamma = gamma
        self.time_horizon = time_horizon
        self.initial_salary_tier = initial_salary_tier
        self.work_location = work_location
        self.gym_location = gym_location
        self.rng = np.random.RandomState(random_seed)

        # Salary tiers (annual)
        self.salaries = {0: 60000, 1: 80000, 2: 100000}

        # Monthly rent by location tier
        self.rents = {0: 1200, 1: 2000, 2: 2800}

        # Reward weights (utility function parameters)
        self.alpha_strength = 5.0
        self.alpha_energy = 2.0
        self.alpha_location = 1.5
        self.alpha_wealth = 2.0
        self.alpha_consumption = 1.0
        self.alpha_stress = 10.0

    def actions(self, state: NonTerminal[StateV3]) -> Iterable[ActionV3]:
        """Return all valid actions for the given state.

        Some actions may be invalid (e.g., can't downgrade from location 0,
        can't upgrade from location 2).
        """
        s = state.state

        # Housing actions constrained by current location
        housing_choices = []
        if s.location > 0:
            housing_choices.append(-1)  # Can downgrade
        housing_choices.append(0)  # Can stay
        if s.location < 2:
            housing_choices.append(1)  # Can upgrade

        # Generate all valid action combinations
        actions = []
        for housing in housing_choices:
            for invest in [0, 1, 2]:
                for training in [0, 1, 2]:
                    for consumption in [0, 1, 2]:
                        actions.append(ActionV3(
                            housing=housing,
                            invest=invest,
                            training=training,
                            consumption=consumption
                        ))

        return actions

    def step(
        self,
        state: NonTerminal[StateV3],
        action: ActionV3
    ) -> Distribution[Tuple[State[StateV3], float]]:
        """
        Execute one step of the MDP, returning a distribution over
        (next_state, reward) pairs.

        This implements the transition dynamics P(s', r | s, a) as a
        sampling-based distribution.
        """
        s = state.state

        def sample_next_state_reward() -> Tuple[State[StateV3], float]:
            """Sample a transition from the current state and action"""

            # Check for terminal condition
            if s.time >= self.time_horizon - 1:
                # Terminal state
                reward = self._compute_reward(s, action) + self._terminal_reward(s)
                return Terminal(s), reward

            # --- Deterministic transitions ---
            new_location = np.clip(s.location + action.housing, 0, 2)
            new_time = s.time + 1

            # Wealth transition (simplified)
            monthly_salary = self.salaries[s.salary] / 12
            rent = self.rents[new_location]
            investment = monthly_salary * action.invest * 0.10
            consumption_cost = 200 + action.consumption * 300  # $200-800/month

            # Expected market return
            market_return = 0.01 if s.market == 1 else -0.005  # Bull/bear monthly return

            wealth_change = int((monthly_salary - rent - investment - consumption_cost +
                               market_return * s.wealth * 10000 / 10000))
            new_wealth = np.clip(s.wealth + wealth_change, 0, 10)

            # --- Stochastic transitions (simplified multinomial) ---

            # Work intensity transition (exogenous AR(1) process)
            # Mean-reverting around intensity=1 (medium)
            work_intensity_probs = self._work_intensity_transition(s.work_intensity)
            new_work_intensity = self.rng.choice([0, 1, 2], p=work_intensity_probs)

            # Energy transition depends on work, training, and location
            energy_probs = self._energy_transition(
                s.energy, new_work_intensity, action.training, new_location
            )
            new_energy = self.rng.choice([0, 1, 2], p=energy_probs)

            # Strength transition depends on training and energy
            strength_levels, strength_probs = self._strength_transition(
                s.strength, action.training, new_energy
            )
            new_strength = self.rng.choice(strength_levels, p=strength_probs)

            # Salary transition (promotion or annual raise)
            salary_probs = self._salary_transition(s.salary, s.time, new_work_intensity)
            new_salary = self.rng.choice([0, 1, 2], p=salary_probs)

            # Market transition (persistent regime)
            market_probs = self._market_transition(s.market)
            new_market = self.rng.choice([0, 1], p=market_probs)

            # Emergency fund (simple rule-based)
            new_emergency = min(2, s.emergency + (1 if new_wealth > 5 else 0))

            # Create next state
            next_state_obj = StateV3(
                wealth=new_wealth,
                emergency=new_emergency,
                location=new_location,
                work_intensity=new_work_intensity,
                energy=new_energy,
                strength=new_strength,
                salary=new_salary,
                market=new_market,
                time=new_time
            )

            # Compute reward
            reward = self._compute_reward(s, action)

            return NonTerminal(next_state_obj), reward

        return SampledDistribution(sample_next_state_reward)

    def _work_intensity_transition(self, current: int) -> np.ndarray:
        """Mean-reverting AR(1) process for work intensity (exogenous)"""
        # Higher persistence, mean-reversion to medium (1)
        if current == 0:  # Low
            return np.array([0.5, 0.4, 0.1])
        elif current == 1:  # Medium
            return np.array([0.2, 0.6, 0.2])
        else:  # High
            return np.array([0.1, 0.4, 0.5])

    def _energy_transition(
        self, current_energy: int, work_intensity: int,
        training: int, location: int
    ) -> np.ndarray:
        """Energy depleted by work and training, boosted by good location (short commute)"""
        # Base probability: stay same
        probs = np.array([0.2, 0.6, 0.2])

        # High work intensity and high training deplete energy
        depletion = work_intensity + training
        if depletion >= 4:  # Both high
            probs = np.array([0.7, 0.25, 0.05])
        elif depletion >= 3:
            probs = np.array([0.4, 0.4, 0.2])

        # Expensive location (short commute) provides energy boost
        if location == 2:  # Expensive/close
            probs += np.array([-0.1, 0.0, 0.1])

        # Normalize and clip
        probs = np.clip(probs, 0.01, 0.99)
        probs /= probs.sum()

        # Shift based on current energy
        if current_energy == 0:  # Low energy, hard to reach high
            return np.array([0.6, 0.3, 0.1]) * probs / np.dot([0.6, 0.3, 0.1], probs)
        elif current_energy == 2:  # High energy
            return np.array([0.1, 0.3, 0.6]) * probs / np.dot([0.1, 0.3, 0.6], probs)

        return probs

    def _strength_transition(
        self, current: int, training: int, energy: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Strength gains from training, modulated by available energy.
        Returns (valid_levels, probabilities) for sampling.
        """
        # Can only gain/lose one level per month
        if current == 0:  # Beginner
            valid_next = np.array([0, 1])
            if training >= 2 and energy >= 1:
                probs = np.array([0.7, 0.3])  # Good chance to advance
            else:
                probs = np.array([0.9, 0.1])  # Mostly stay
        elif current == 1:  # Intermediate
            valid_next = np.array([0, 1, 2])
            if training >= 2 and energy >= 2:
                probs = np.array([0.0, 0.7, 0.3])  # Can advance
            elif training == 0:
                probs = np.array([0.1, 0.8, 0.1])  # Might regress
            else:
                probs = np.array([0.0, 0.9, 0.1])  # Mostly maintain
        else:  # Advanced (current == 2)
            valid_next = np.array([1, 2])
            if training >= 2 and energy >= 2:
                probs = np.array([0.0, 1.0])  # Maintain
            else:
                probs = np.array([0.2, 0.8])  # Risk regression

        return valid_next, probs

    def _salary_transition(
        self, current: int, time: int, avg_work_intensity: int
    ) -> np.ndarray:
        """Salary transitions via promotion (depends on tenure and work intensity)"""
        probs = np.zeros(3)

        if current == 0:  # $60k
            # Promotion probability increases with time and work intensity
            promo_prob = 0.0
            if time >= 24:  # After 2 years
                promo_prob = 0.15 + 0.05 * avg_work_intensity
            probs[0] = 1 - promo_prob
            probs[1] = promo_prob
        elif current == 1:  # $80k
            promo_prob = 0.0
            if time >= 30:  # After 2.5 years
                promo_prob = 0.10 + 0.03 * avg_work_intensity
            probs[1] = 1 - promo_prob
            probs[2] = promo_prob
        else:  # $100k+
            probs[2] = 1.0  # Can't advance further

        return probs

    def _market_transition(self, current: int) -> np.ndarray:
        """Two-state Markov chain with persistence"""
        if current == 0:  # Bear
            return np.array([0.6, 0.4])  # 60% stay bear
        else:  # Bull
            return np.array([0.2, 0.8])  # 80% stay bull

    def _compute_reward(self, state: StateV3, action: ActionV3) -> float:
        """
        Compute immediate reward from state and action.

        Multi-objective utility with components:
        - Strength (training performance is primary goal)
        - Energy (feeling energized)
        - Location quality (amenities, walkability)
        - Wealth security (diminishing returns)
        - Consumption enjoyment
        - Stress penalty (low wealth + no emergency fund)
        """
        reward = 0.0

        # 1. Strength utility (training performance)
        reward += self.alpha_strength * state.strength

        # 2. Energy utility
        reward += self.alpha_energy * state.energy

        # 3. Location quality
        reward += self.alpha_location * state.location

        # 4. Wealth security (square root for diminishing returns)
        reward += self.alpha_wealth * np.sqrt(state.wealth)

        # 5. Consumption enjoyment
        reward += self.alpha_consumption * action.consumption

        # 6. Stress penalty (low wealth AND no emergency fund)
        if state.wealth < 3 and state.emergency < 2:
            reward -= self.alpha_stress

        return reward

    def _terminal_reward(self, state: StateV3) -> float:
        """Terminal reward for final wealth and strength"""
        return 10.0 * state.wealth + 5.0 * state.strength


# ============================================================================
# Example usage and testing
# ============================================================================

def example_usage():
    """Demonstrate MDP instantiation and basic usage"""

    # Create MDP instance (V3 version for Phase 2)
    mdp = MultiCapitalEnergyMDP(
        version='v3',
        gamma=0.99,
        time_horizon=36,
        random_seed=42
    )

    # Create an initial state
    initial_state = NonTerminal(StateV3(
        wealth=5,  # $50k liquid
        emergency=1,  # Partial emergency fund
        location=1,  # Medium neighborhood
        work_intensity=1,  # Medium work intensity
        energy=2,  # High energy (just started)
        strength=0,  # Beginner strength
        salary=0,  # $60k salary
        market=1,  # Bull market
        time=0  # Start of horizon
    ))

    # Get available actions
    actions = list(mdp.actions(initial_state))
    print(f"Number of available actions from initial state: {len(actions)}")
    print(f"Example action: {actions[0]}")

    # Sample a transition
    action = actions[0]
    next_dist = mdp.step(initial_state, action)
    next_state, reward = next_dist.sample()

    print(f"\nInitial state: {initial_state.state}")
    print(f"Action taken: {action}")
    print(f"Next state: {next_state.state if isinstance(next_state, NonTerminal) else 'Terminal'}")
    print(f"Reward: {reward:.2f}")

    print(f"\nMDP configured for version: {mdp.version}")
    print(f"Discount factor: {mdp.gamma}")
    print(f"Time horizon: {mdp.time_horizon} months")

    # State space size estimate
    state_space_size = 11 * 3 * 3 * 3 * 3 * 3 * 3 * 2 * 36
    print(f"\nEstimated state space size: {state_space_size:,} states")
    print("This is tractable with exact dynamic programming!")


if __name__ == '__main__':
    example_usage()
